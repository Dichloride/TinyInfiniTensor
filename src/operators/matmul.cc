#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        IT_ASSERT(inputs.size() == 2);
        const auto A = inputs[0];
        const auto B = inputs[1];
        auto aDims = A->getDims();
        auto bDims = B->getDims();
        IT_ASSERT(aDims.size() >= 2);
        IT_ASSERT(bDims.size() >= 2);

        int aRank = static_cast<int>(aDims.size());
        int bRank = static_cast<int>(bDims.size());

        // Matrix dimensions (m x k) * (k x n).
        int aM = transA ? aDims[aRank - 1] : aDims[aRank - 2];
        int aK = transA ? aDims[aRank - 2] : aDims[aRank - 1];
        int bK = transB ? bDims[bRank - 1] : bDims[bRank - 2];
        int bN = transB ? bDims[bRank - 2] : bDims[bRank - 1];

        IT_ASSERT(aK == bK);
        m = aM;
        n = bN;
        k = aK;

        Shape aBatch(aDims.begin(), aDims.end() - 2);
        Shape bBatch(bDims.begin(), bDims.end() - 2);
        Shape batch = infer_broadcast(aBatch, bBatch);

        Shape out = batch;
        out.push_back(m);
        out.push_back(n);
        return {{out}};
    }

} // namespace infini
