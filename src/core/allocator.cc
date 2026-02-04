#include "core/allocator.h"
#include <algorithm>
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        // First-fit allocation from free blocks.
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it)
        {
            auto addr = it->first;
            auto blockSize = it->second;
            if (blockSize < size)
                continue;

            // Allocate from the beginning of this free block.
            freeBlocks.erase(it);
            if (blockSize > size)
            {
                freeBlocks.emplace(addr + size, blockSize - size);
            }
            return addr;
        }

        // Allocate from the end of the arena.
        auto addr = used;
        used += size;
        peak = std::max(peak, used);
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        if (size == 0)
            return;

        // Insert the freed block and coalesce with neighbors.
        auto it = freeBlocks.lower_bound(addr);

        // Merge with previous if adjacent.
        if (it != freeBlocks.begin())
        {
            auto pit = std::prev(it);
            if (pit->first + pit->second == addr)
            {
                addr = pit->first;
                size += pit->second;
                freeBlocks.erase(pit);
            }
        }

        // Merge with next if adjacent.
        if (it != freeBlocks.end() && addr + size == it->first)
        {
            size += it->second;
            freeBlocks.erase(it);
        }

        freeBlocks.emplace(addr, size);

        // If the last part of the arena becomes free, shrink `used` and
        // repeatedly trim contiguous free blocks at the end.
        while (true)
        {
            auto last = freeBlocks.empty() ? freeBlocks.end()
                                           : std::prev(freeBlocks.end());
            if (last == freeBlocks.end())
                break;
            if (last->first + last->second != used)
                break;
            used = last->first;
            freeBlocks.erase(last);
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
