#pragma once

#include <stdio.h>
#include <algorithm>
#include <thread>
#include <functional>
#include <vector>
#include <atomic>
#include <mutex>


namespace
{
    void parallel_for_range(
        uint64_t item_start,
        uint64_t item_stop,
        std::function<void (uint64_t item)> functor_background,
        std::function<uint64_t (uint64_t item)> functor_foreground = nullptr,
        std::function<bool (uint64_t item)> functor_done = nullptr)
    {
        std::atomic<uint64_t> atomic_count(item_start);
        std::mutex mutex;

        unsigned processor_count = std::thread::hardware_concurrency();
        unsigned thread_count = processor_count == 0 ? 8 : processor_count;
        printf("parallel_for_range using %u threads\n", thread_count);
        std::vector<std::thread> threads(thread_count);
        for(size_t i = 0; i < thread_count; ++i)
        {
            threads[i] = std::thread([&, item_stop]()
            {
                uint64_t item;
                while ((item = atomic_count++) < item_stop)
                {
                    functor_background(item);

                    if (functor_done)
                    {
                        std::lock_guard<std::mutex> lock (mutex);
                        (void) lock; // prevent the "unused variable" warning
                        if (!functor_done(item))
                            return;
                    }
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

                if (functor_done)
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    (void) lock; // prevent the "unused variable" warning
                    if (!functor_done(item))
                        break;
                }
            }
        }

        for (std::thread& thread: threads)
            thread.join();
    }

} // anonymous namespace
