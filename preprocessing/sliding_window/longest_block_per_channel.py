def find_longest_block_per_channel(contiguous_windows):

    longest = {}

    for trial, channels in contiguous_windows.items():

        for channel, ranges in channels.items():

            for start, end in ranges:

                length = end - start + 1

                if (
                    channel not in longest
                    or
                    length > longest[channel]["length"]
                ):

                    longest[channel] = {
                        "trial": trial,
                        "start": start,
                        "end": end,
                        "length": length
                    }

    return longest