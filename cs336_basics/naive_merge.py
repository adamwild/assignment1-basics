def naive_merge(s_frequency_table, merge_pair):
    """_summary_

    Args:
        s_frequency_table (dict[tuple[bytes], int]): {(b'l', b'o', b'w'): 1, (b' ', b'l', b'o', b'w'): 4,
        merge_pair (tuple[bytes]): (b's', b't')
    """
    new_s_frequency_table = {}
    new_byte_pair_count = {}
    
    b1_target, b2_target = merge_pair
    merged_token = b1_target + b2_target

    for s_token, count in s_frequency_table.items():
        # Handle single-byte tokens immediately
        if len(s_token) == 1:
            # new_byte_pair_count[s_token] = new_byte_pair_count.get(s_token, 0) + count
            continue

        if len(s_token) == 2:
            b1, b2 = s_token

            if (b1, b2) == merge_pair:
                # new_byte_pair_count[merged_token] = new_byte_pair_count.get(merged_token, 0) + count
                # new_s_frequency_table[merged_token] = new_s_frequency_table.get(merged_token, 0) + count
                pass
            else:
                new_byte_pair_count[(b1, b2)] = new_byte_pair_count.get((b1, b2), 0) + count
                new_s_frequency_table[(b1, b2)] = new_s_frequency_table.get((b1, b2), 0) + count
            continue

        # 3 or more bytes
        new_s_token = []

        ind_byte = 0

        # If b1 belongs to a merged pair from his predecessor, consider b1 as the merged bytes
        rename_b1 = False

        while ind_byte < len(s_token)-1:
            b1, b2 = s_token[ind_byte], s_token[ind_byte+1]

            if rename_b1:
                b1 = merged_token
                rename_b1 = False

            # Last two bytes
            if ind_byte == len(s_token)-2:
                new_byte_pair_count[(b1, b2)] = new_byte_pair_count.get((b1, b2), 0) + count
                new_s_token.append(b1)
                new_s_token.append(b2)
                break

            b3 = s_token[ind_byte+2]

            # We write the possible cases, b_i means a byte that doesn't merge, c_i is a byte that belongs to a mergure
            if (b1, b2) != merge_pair:
                new_s_token.append(b1)

                # b1, b2, b3
                if (b2, b3) != merge_pair:
                    new_byte_pair_count[(b1, b2)] = new_byte_pair_count.get((b1, b2), 0) + count
                    ind_byte += 1

                # b1, c2, c3
                else:
                    new_byte_pair_count[(b1, merged_token)] = new_byte_pair_count.get((b1, merged_token), 0) + count
                    ind_byte += 2

                    # We restart the loop on c3 which belongs to a merge
                    rename_b1 = True

                    # c2, c3 is the end of the token
                    if ind_byte == len(s_token)-1:
                        new_s_token.append(merged_token)

            else:
                new_s_token.append(merged_token)

                if ind_byte + 3 <= len(s_token)-1:
                    b4 = s_token[ind_byte+3]

                    # c1, c2, b3, b4
                    if (b3, b4) != merge_pair:
                        new_byte_pair_count[(merged_token, b3)] = new_byte_pair_count.get((merged_token, b3), 0) + count
                        ind_byte += 2

                    # c1, c2, c3, c4
                    else:
                        new_byte_pair_count[(merged_token, merged_token)] = new_byte_pair_count.get((merged_token, merged_token), 0) + count
                        ind_byte += 3
                        rename_b1 = True

                # c1, c2, b3]
                else:
                    new_byte_pair_count[(merged_token, b3)] = new_byte_pair_count.get((merged_token, b3), 0) + count
                    ind_byte += 2
                    new_s_token.append(b3)

        new_s_frequency_table[tuple(new_s_token)] = new_s_frequency_table.get(tuple(new_s_token), 0) + count

    return new_s_frequency_table, new_byte_pair_count

def get_next_merge(byte_pair_count):
    # Next pair to merge by finding pairs with highest counts, ties are broken lexicographically
    max_count = max(count for count in byte_pair_count.values())
    tied_first_place = [byte_pair for byte_pair, count in byte_pair_count.items() if count == max_count]
    next_pair_merge = max(tied_first_place)
    
    return next_pair_merge


def naive_merge_loop(vocab, merges, s_frequency_table, byte_pair_count):
    new_merge = get_next_merge(byte_pair_count)
    merges.append(new_merge)

    s_frequency_table, byte_pair_count = naive_merge(s_frequency_table, new_merge)

    vocab[len(vocab)] = new_merge[0] + new_merge[1]

    return vocab, merges, s_frequency_table, byte_pair_count