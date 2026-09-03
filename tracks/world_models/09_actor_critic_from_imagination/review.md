# Review checklist

- Can you derive each trajectory weight from the preceding discounts?
- Does the actor minimize the negative weighted imagined return?
- Do actor gradients pass through world-model computations without training it?
- Are critic targets and critic input states detached?
- Why are value predictions recomputed instead of using `imagined.values`?
- Does actor backward leave critic parameter gradients empty?
- Does value backward leave actor parameter gradients empty?
- Which stability features from larger Dreamer systems are intentionally absent?
