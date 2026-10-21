# Notes

What surprised me, one entry per surprise: what I expected, what happened, my guess at why.


## Questions

1. q_proj + v_proj what is it ?
2. what is the low rank ? Explain in high (abstract, whu we need, analogies, comparison etc) and low level (exactly what is it, why)
3. What does it mean Lora Does NOT save: activations ?
4. Look at the section below (I found this exciting), 1 point, for me it looks similar with PCA or SVD ideas from linear algebra. Is there a connection or analogy ?
5. I didn't quite understand how exactly alpha / r "keeps the update's size roughly stable when you change r, so you need not retune the learning rate each time"
6. I read the section 1 in defense.md. Why are you benchmarking with tf-idf only and not with not fine tuned version ? I also want to know how the fine tune helped.
7. What is the difference between fp32 and int8 ? Why are we using int not float ? Can we do so ?
8. "per-channel: one ruler per output row" what is the output shape and its interpretation: what the row means ? The column ?
9. Why you used train/test split and not train/val/test split ?

## I found this exciting

1. 
"Why "Low" Rank Works
In linear algebra, the rank of a matrix represents the number of independent, unique directions of information it contains.
Giant models have a high intrinsic rank (lots of complex concepts stored). However, when you fine-tune a model on a specific task or style, the change in knowledge has a low intrinsic rank. The model doesn't need to rebuild its entire brain; it just needs a few specific tweaks, which can easily be captured by those skinny, low-rank matrices."
I love this explanation above, I seem to understand the main idea.


## My explanations of the reading.md file

1. The Lora section I see as follows: imagine we have a pretrained LLM, researchers and other guys spent a lot of resources to train it and they end well I guess. Now I have a very specific task, I start with a baseline, the aforementioned pretrained LLM. It performs ok but i believe the results could be better if it would know the domain better. But we don't need to train the full model, because it already speaks the correct language, it knows how the language mechanics work, it only need a few tweaks to grasp new ideas from a specific domain. So in the spirit of linear algebra, we won't train the whole model but only those necessary tweaks. To be exact the low rank of query and value projections. So now we write y as Wx + (r / alpha) * B @ Ax. And we train only A and B which are 0.5% of total weights.

2. The quantization is like approximation or ruler with discrete values. There several ways to quantize: post training, during training (which improves the results bcs the model during training teaches how to leave with quantization). There are per-tensor (whole weight matrix) and per-channel (for each channel its own quantization) quantization. Also during the quantization it is import to keep track of the outliers, bcs one outlier can mess up the whole quantization. Ideas can be to store outliers separately.

3. We need to measure the results in the right way, thoughtfully. For example in our case, the difference between fp32 and int8 didn't show signifficant difference, which is nice I guess. Also with quantization it is important to look for a rare class.

4. Self-test.
    1. I guess we start with zero B bcs than the training starts smooth, bcs simply y = Wx, but if we start with random B then at first model will start to learn to completely other direction. I cannot explain it better, just an intuition
    2. Bcs still 0.1% of weights is still a lot. But I guess it is not correct answer. Idk
    3. I blindly remembered that it smoothes the training, but I didn't understand why. I asked it as a question above.
    4. I would be honest I needed to peek (google what is merged and unmerged). I understood that merged is when you update the original model weights tensor by a factor of : alpha / r * BA. But I didn;t get what are pros of un merged and why not simply do merged which do not changes how we use the model later.
    5. One outlier would wuin the per-tensor bcs it ruins the whole matrix, while per-channel outlier would ruin only the corresponding output channel.
    6. Tbh I didn't understand. Static is when you quantize one time only or what ?
    7. Tbh I don't understand the question bcs what else can be quantized if not weights ? The activation ? Then I don't understand what it means to quantize activations. 
    8. From the text I remember something about CPU in mac, but it was too specific for me, I am not a computer science geek with 40 years of experience starting from the time when 1 kb of memory took the space of a house. Haha. My best answer is that: yes, by quantization we made the model less memory but the calculation for getting the prediction still takes time, bcs we just made weights use less memory but we did not remove some weights, the cardinality is still the same.
    9. It depends on the test after. If it is significant, then yes. But also i belive drop in the performance can be ok depending on the other gains (e.g. previously we was not able to run the model on the phone, but now with a little decline in accuracy, but we can)