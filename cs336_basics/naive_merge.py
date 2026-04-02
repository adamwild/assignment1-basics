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

def reduce_s_token(token_id, merge_pair, id_token_count, byte_pair_count, byte_pair_index):
    """From a s_token with at least one merge_pair, compute the resulting merged s_token
    Updates relevant indexes

    Args:
        token_id (int): Id of the token to apply merge to
        merge_pair (tuple[bytes]): A pair of bytes as basis for mergure
                                ex: (b'e', b'r')
        id_token_count (dict[int][list[bytes], int]): A dictionary linking token_id to the current pre-token and its related count
                                ex: {168: [[b' ', b'l', b'ow'], 7]}
        byte_pair_count (dict[tuple[bytes]]=int): Dictionary of current bytes pair count
                                ex: {(b1, b2): 3}
        byte_pair_index (dict[tuple[bytes]]=list[int]): Dictionary linking the pair of bytes to the pre-token where they appear (can appear several times)
                                ex: {(b1, b2): [168, 94]}
    """
    mp1, mp2 = merge_pair
    merged_bytes = mp1 + mp2

    # s_token (tuple[bytes]): A single pre-token with current vocab
    #                         ex: (b' ', b'l', b'ow', b'e', b'r')
    # token_count (int): Number of time this pre-token appears in total in the training text
    s_token, token_count = id_token_count[token_id]
    new_s_token = []

    ind_byte = 0
    has_merged = False
    while ind_byte < len(s_token):

        # b1]
        if ind_byte == len(s_token)-1:
            b1, b2 = s_token[ind_byte], None
        else:
            b1, b2 = s_token[ind_byte], s_token[ind_byte+1]

        # b1, cb2
        if (b1, b2) != merge_pair:
            new_s_token.append(b1)
            ind_byte += 1

            # m2, b1, cb2
            if has_merged:
                byte_pair_count[(merged_bytes, b1)] = byte_pair_count.get((merged_bytes, b1), 0) + token_count
                byte_pair_index.setdefault((merged_bytes, b1), []).append(token_id)

                byte_pair_count[(s_token[ind_byte-2], b1)] -= token_count
                byte_pair_index[(s_token[ind_byte-2], b1)].remove(token_id)

            has_merged = False

            # b1, b2
            if b2 and b2 != mp1:
                new_s_token.append(b2)
                ind_byte += 1

        # (b0), c1, c2, (b3)
        else:
            
            # b0, c1, c2
            if new_s_token:
                # Could either be the original previous bytes or a fresh mergure
                b0 = new_s_token[-1]
                

                # If b0 was a previous merge (i.e. m0), the pair mp2, mp1 must be decreased
                if has_merged:
                    byte_pair_count[(merged_bytes, merged_bytes)] = byte_pair_count.get((merged_bytes, merged_bytes), 0) + token_count
                    byte_pair_index.setdefault((merged_bytes, merged_bytes), []).append(token_id)

                    byte_pair_count[(mp2, mp1)] -= token_count
                    byte_pair_index[(mp2, mp1)].remove(token_id)

                # Previous byte was a normal one, handle b0, mp
                else:
                    byte_pair_count[(b0, merged_bytes)] = byte_pair_count.get((b0, merged_bytes), 0) + token_count
                    byte_pair_index.setdefault((b0, merged_bytes), []).append(token_id)

                    byte_pair_count[(b0, b1)] -= token_count
                    byte_pair_index[(b0, b1)].remove(token_id)

            ind_byte += 2

            new_s_token.append(merged_bytes)
            has_merged = True

    id_token_count[token_id][0] = new_s_token

    return id_token_count, byte_pair_count, byte_pair_index