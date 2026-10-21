# Notes

What surprised me, one entry per surprise: what I expected, what happened, my guess at why.


## Questions

1. q_proj + v_proj what is it ?
2. what is the low rank ? Explain in high (abstract, whu we need, analogies, comparison etc) and low level (exactly what is it, why)
3. What does it mean Lora Does NOT save: activations ?

## I found this exciting

1. 
Why "Low" Rank Works

In linear algebra, the rank of a matrix represents the number of independent, unique directions of information it contains.
Giant models have a high intrinsic rank (lots of complex concepts stored). However, when you fine-tune a model on a specific task or style, the change in knowledge has a low intrinsic rank. The model doesn't need to rebuild its entire brain; it just needs a few specific tweaks, which can easily be captured by those skinny, low-rank matrices.
