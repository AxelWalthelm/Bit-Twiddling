#pragma once

#include <stdio.h>
#include <algorithm>
#include <thread>
#include <functional>
#include <vector>
#include <atomic>


namespace
{
    void parallel_for_range(
        uint64_t item_start,
        uint64_t item_stop,
        std::function<void (uint64_t item)> functor_background,
        std::function<uint64_t (uint64_t item)> functor_foreground = nullptr)
    {
        std::atomic<uint64_t> atomic_count(item_start);

        unsigned processor_count = std::thread::hardware_concurrency();
        unsigned thread_count = processor_count == 0 ? 8 : processor_count;
        printf("parallel_for_range using %u threads\n", thread_count);
        std::vector<std::thread> threads(thread_count);
        for(size_t i = 0; i < thread_count; ++i)
        {
            threads[i] = std::thread([&atomic_count, item_stop, &functor_background]()
            {
                uint64_t item;
                while ((item = atomic_count++) < item_stop)
                {
                    functor_background(item);
                }
            });
        }

        if (functor_foreground)
        {
            uint64_t item;
            while ((item = atomic_count++) < item_stop)
            {
                uint64_t new_item = functor_foreground(item);

                uint64_t current_item = atomic_count;
                while(current_item < new_item && !atomic_count.compare_exchange_weak(current_item, new_item))
                    ;
            }
        }

        for (std::thread& thread: threads)
            thread.join();
    }

} // anonymous namespace
