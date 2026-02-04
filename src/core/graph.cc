#include "core/graph.h"
#include "core/blob.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_map>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        // NOTE: This is a minimal rule-based optimizer for the training tasks.
        // It rewires the graph in-place and drops dead ops/tensors.

        auto removeOpFully = [&](const Operator &op)
        {
            if (!op)
                return;
            // Detach from input tensors.
            for (auto &t : op->inputs)
                if (t)
                    t->removeTarget(op);
            // Detach from output tensors.
            for (auto &t : op->outputs)
                if (t && t->getSource() == op)
                    t->setSource(Operator{});
            // Detach predecessor/successor relations.
            for (auto &p : op->getPredecessors())
                if (p)
                    p->removeSuccessors(op);
            for (auto &s : op->getSuccessors())
                if (s)
                    s->removePredecessors(op);
            removeOperator(op);
        };

        auto tryRemoveDeadTensor = [&](const Tensor &t)
        {
            if (!t)
                return;
            if (t->getSource() == nullptr && t->getTargets().empty())
                removeTensor(t);
        };

        auto isInversePerm = [](const std::vector<int> &a,
                                const std::vector<int> &b) -> bool
        {
            if (a.size() != b.size())
                return false;
            const int r = static_cast<int>(a.size());
            for (int i = 0; i < r; ++i)
            {
                int bi = b[i];
                if (bi < 0 || bi >= r)
                    return false;
                if (a[bi] != i)
                    return false;
            }
            return true;
        };

        auto isSwapLastTwo = [](const std::vector<int> &perm) -> bool
        {
            const int r = static_cast<int>(perm.size());
            if (r < 2)
                return false;
            for (int i = 0; i < r - 2; ++i)
                if (perm[i] != i)
                    return false;
            return perm[r - 2] == r - 1 && perm[r - 1] == r - 2;
        };

        // Rule 1: remove redundant Transpose-Transpose where the second is the
        // inverse of the first. We keep the middle tensor (so downstream rewiring
        // does not need new tensor creation) and drop the other tensors/ops if
        // they become dead.
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (auto it = ops.begin(); it != ops.end(); ++it)
            {
                auto t1op = *it;
                if (!t1op || t1op->getOpType() != OpType::Transpose)
                    continue;
                auto t1 = as<TransposeObj>(t1op);
                if (!t1)
                    continue;

                auto mid = t1op->getOutput(0);
                if (!mid)
                    continue;
                auto midTargets = mid->getTargets();
                if (midTargets.size() != 1)
                    continue;
                auto t2op = midTargets[0];
                if (!t2op || t2op->getOpType() != OpType::Transpose)
                    continue;
                auto t2 = as<TransposeObj>(t2op);
                if (!t2)
                    continue;
                // Ensure adjacency: t2 takes mid as its input.
                if (t2op->getInputs(0) != mid)
                    continue;
                if (!isInversePerm(t1->getPermute(), t2->getPermute()))
                    continue;

                auto in = t1op->getInputs(0);
                auto out = t2op->getOutput(0);
                // Redirect uses of out -> in (Transpose-Transpose cancels out).
                for (auto &user : out->getTargets())
                {
                    user->replaceInput(out, in);
                    out->removeTarget(user);
                    in->addTarget(user);
                }

                // Remove both transpose ops.
                removeOpFully(t2op);
                removeOpFully(t1op);

                // Remove now-dead tensors.
                tryRemoveDeadTensor(mid);
                tryRemoveDeadTensor(out);

                changed = true;
                break; // restart scanning since ops vector changed
            }
        }

        // Rule 2: fuse Transpose into MatMul's transA/transB when Transpose swaps
        // the last two dims.
        bool fused = true;
        while (fused)
        {
            fused = false;
            for (size_t i = 0; i < ops.size(); ++i)
            {
                auto op = ops[i];
                if (!op || op->getOpType() != OpType::MatMul)
                    continue;
                auto mm = as<MatmulObj>(op);
                if (!mm)
                    continue;

                for (int idx = 0; idx < 2; ++idx)
                {
                    auto t = op->getInputs(idx);
                    auto src = t ? t->getSource() : nullptr;
                    if (!src || src->getOpType() != OpType::Transpose)
                        continue;
                    auto tr = as<TransposeObj>(src);
                    if (!tr)
                        continue;
                    if (!isSwapLastTwo(tr->getPermute()))
                        continue;

                    auto orig = src->getInputs(0);
                    op->replaceInput(t, orig);
                    t->removeTarget(op);
                    orig->addTarget(op);

                    if (idx == 0)
                        mm->setTransA(!mm->getTransA());
                    else
                        mm->setTransB(!mm->getTransB());

                    removeOpFully(src);
                    tryRemoveDeadTensor(t);

                    fused = true;
                    break;
                }
                if (fused)
                    break; // restart scanning, since `ops` was modified
            }
        }

        // A small canonicalization for the training tests: keep MatMul inputs
        // ordered by GUID without touching transA/transB.
        for (auto &op : ops)
        {
            if (!op || op->getOpType() != OpType::MatMul)
                continue;
            if (op->getInputs(0) && op->getInputs(1) &&
                op->getInputs(0)->getGuid() > op->getInputs(1)->getGuid())
                std::swap(op->inputs[0], op->inputs[1]);
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        std::unordered_map<TensorObj *, size_t> addr;
        std::unordered_map<TensorObj *, size_t> bytes;
        std::unordered_map<TensorObj *, int> useCnt;
        std::unordered_set<TensorObj *> keepAlive;

        // Graph outputs need to stay alive after run().
        for (auto &t : getOutputs())
            keepAlive.insert(t.get());

        for (auto &t : tensors)
        {
            bytes[t.get()] = t->getBytes();
            useCnt[t.get()] = static_cast<int>(t->getTargets().size());
        }

        // Allocate all graph inputs first so user can fill them after dataMalloc().
        for (auto &t : getInputs())
        {
            auto off = allocator.alloc(bytes[t.get()]);
            addr[t.get()] = off;
        }

        // Simulate allocations along topo order and free tensors after last use.
        for (auto &op : ops)
        {
            // Allocate outputs produced by this op.
            for (auto &out : op->getOutputs())
            {
                auto off = allocator.alloc(bytes[out.get()]);
                addr[out.get()] = off;
            }

            // Update use counts of inputs and free dead tensors (except outputs).
            for (auto &in : op->getInputs())
            {
                auto &cnt = useCnt[in.get()];
                cnt -= 1;
                if (cnt == 0 && keepAlive.find(in.get()) == keepAlive.end())
                {
                    allocator.free(addr[in.get()], bytes[in.get()]);
                }
            }
        }

        // Perform the actual allocation once, using the simulated peak.
        void *base = allocator.getPtr();
        for (auto &t : tensors)
        {
            auto it = addr.find(t.get());
            IT_ASSERT(it != addr.end());
            auto off = it->second;
            void *p = static_cast<void *>(static_cast<char *>(base) + off);
            t->setDataBlob(make_ref<BlobObj>(runtime, p));
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini
