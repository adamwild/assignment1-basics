def get_next_merge(byte_pair_count):
    # Next pair to merge by finding pairs with highest counts, ties are broken lexicographically
    max_count = max(count for count in byte_pair_count.values())
    tied_first_place = [byte_pair for byte_pair, count in byte_pair_count.items() if count == max_count]
    next_pair_merge = max(tied_first_place)
    
    return next_pair_merge

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