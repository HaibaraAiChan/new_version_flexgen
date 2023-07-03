We aim to achieve model/pipeline/sequence parallelism in OPT models. To simplify this objective, our plan is to integrate flexgen offloading with parallelism in Megatron.

Currently, we have extracted the code we need from Megatron and Flexgen.
The framework of our design is denoted below.



### Current files structure
#### model/:    
            all layers defined , which are used in OPT model. E.g. input layer, MLP_layer, self_attention_layer, output_layer     
#### flexgen_additional/:  
            flexgen related files (computation implementation etc.)   
#### mpu/: 
            parallelism related dependent files from megatron 
#### tokenizer/:  
            tokenizer dependent files from megatron 
#### examples/: 
            our test files
#### dist_signal_handler.py : 
            dependent file from megatron 
#### microbatches.py : 
            dependent file from megatron   


The code we need to modify are mainly located in folder model/ and flexgen_additional/.
For example, in ```model/self_attention_layer.py ``` the multi-head attention computation in prefill phase is implemented by ```function mha() ```in   ```flexgen_additional/pytorch_backend.py``` .  

   

Now, we are working on the ``` model/fused_layer_norm.py  ``` replace the original ```torch.nn.functional.layer_norm``` in ```model/self_attention_layer.py```.   
The target of this operation is utilizing ```fused_layer_norm``` to help ```self-Attention layer``` implement sequence parallelism.  


Later, we will add other parallelism to OPT model based on the folder``` mpu/```.
