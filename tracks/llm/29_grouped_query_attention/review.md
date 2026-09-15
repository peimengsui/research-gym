# Review checklist

- Is `num_query_heads` divisible by `num_kv_heads`?
- Does each KV head map to consecutive query heads in its group?
- Are compact KV tensors expanded only for attention computation?
- Does equal query/KV head count exactly recover multi-head attention?
- Does one KV head recover the multi-query attention layout?
- Are key/value projection widths `num_kv_heads * head_dim`?
- Are query and compact keys rotated at the same absolute positions?
- Does the persistent cache retain `num_kv_heads`, not `num_query_heads`?
- Does cache accounting include both keys and values?
- Is cache reduction proportional to `num_query_heads / num_kv_heads`?
- Do full-context and incremental GQA outputs match?
- What expressiveness-versus-memory tradeoff does KV sharing introduce?
