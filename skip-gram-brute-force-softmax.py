import cupy as cp 
# print(cp.cuda.Device(0).name)

with tqdm(total=total_iterations, desc="Training Skip-gram") as pbar:
    print("जय बजरंग बली") # ॐ 
    for epoch in tqdm(range(epochs)):

        epoch_loss = 0
        # run through the corpus treating each word as a center word
        for c_idx, center_word_id in enumerate(corpus_ids):
            v_c = V[center_word_id, :]
    
            # iterate over the context window to find context|center (o|c) pairs
            # context window would be from [c_idx - m, c_idx + m] excluding c_idx
            # left boundary = max(0, c_idx - m), right boundary = min(corpus_size - 1, c_idx + m)
            context_words = [idx for idx in range(max(0, c_idx - m), min(len(corpus_ids), c_idx + m + 1)) if idx != c_idx]
            
            for o_idx in context_words:
                context_word_id = corpus_ids[o_idx]
                # pair (o|c) -> (o_idx | c_idx)
                u_o = U[context_word_id, :]
                
                # calculate softmax probability -> p(o|c)
                dot_prods = cp.dot(U, v_c) # dot prodcuts of the entire context embedding wrods with the current center word
                max_dot = cp.max(dot_prods) # max of the dot products for the softmax trick
                smx_denom = cp.sum(cp.exp(dot_prods - max_dot)) # using the softmax trick
                curr_dot = cp.dot(u_o, v_c) # dot product of the current center and context words
                smx_num = cp.exp(curr_dot - max_dot)
                p_o_c = smx_num/smx_denom 
    
                # current loss of o and c is the log softmax prob
                # log(smx_num) = curr_dot
                curr_loss = -curr_dot + max_dot + cp.log(cp.sum(cp.exp(dot_prods - max_dot)))
                epoch_loss += curr_loss
    
                # calculate p(w | c)
                p_w_c = cp.exp(dot_prods - max_dot)/smx_denom
                
                # compute the gradients 
                # for v_c (center word) [calculated with the softmax trick]
                grad_v_c = -u_o + cp.dot(p_w_c, U)
    
                # for u_w (other context words excluding the current context word)
                # print(p_w_c.shape, v_c.shape)
                grad_u_w = cp.outer(p_w_c, v_c)
                grad_u_w[context_word_id, :] = 0
    
                # for u_o (current context word)
                grad_u_o = v_c * (p_o_c - 1)

                # update the parameters
                V[center_word_id, :] -= lr * (grad_v_c)
                U -= lr * (grad_u_w)
                U[context_word_id, :] -= lr * (grad_u_o)
            
        avg_epoch_loss = epoch_loss / (len(corpus) * 2 * m)  # rough estimate of pairs
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss}")

        # save the best parameters
        if avg_epoch_loss < best_loss:
            U_best = U.copy()
            V_best = V.copy()
            best_loss = avg_epoch_loss
                
pbar.close()
