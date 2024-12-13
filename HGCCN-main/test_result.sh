
# simplehgn
#MF-DC
python3 DP_AC_D2PT_search_retrain.py --dataset=DBLP -gnn-model=simpleHGN --valid-attributed-type=1 --hidden-dim 64 --num-heads 8 --attn-vec-dim 128 --num_layers 2 --beta_1 1 --droupout 0.2 --slope 0.05 --grad_clip 5 --T 20 --alpha 0.03

#python3 DP_AC_D2PT_search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --valid-attributed-type=0 --hidden-dim 64 --num-heads 8 --attn-vec-dim 128 --num_layers 2 --beta_1 1 --droupout 0.5 --slope 0.1 --grad_clip 5 --T 15 --alpha 0.01

#python3 DP_AC_D2PT_search_retrain.py --dataset=ACM --gnn-model=simpleHGN --valid-attributed-type=0 --hidden-dim 64 --num-heads 8 --attn-vec-dim 128 --num_layers 2 --beta_1 0.8 --droupout 0.2 --slope 0.05 --grad_clip 2 --T 12 --alpha 0.07

# # magnn
#MF-DC
#python3 DP_AC_D2PT_search_retrain.py --dataset=DBLP -gnn-model=magnn --valid-attributed-type=1 --hidden-dim 64 --num-heads 8 --attn-vec-dim 128 --num_layers 2 --complete_num_layers 2 --beta_1 1 --droupout 0.2 --slope 0.05 --grad_clip 5 --T 20 --alpha 0.05

#python3 DP_AC_D2PT_search_retrain.py --dataset=IMDB --gnn-model=magnn --valid-attributed-type=0 --hidden-dim 64 --num-heads 8 --attn-vec-dim 128 --num_layers 3 --complete_num_layers 3 --beta_1 1 --droupout 0.2 --slope 0.1 --grad_clip 5 --T 20 --alpha 0.1

#python3 DP_AC_D2PT_search_retrain.py --dataset=ACM --gnn-model=magnn --valid-attributed-type=0 --hidden-dim 64 --num-heads 8 --attn-vec-dim 128 --num_layers 3 --complete_num_layers 3 --beta_1 0.8 --droupout 0.2 --slope 0.05 --grad_clip 2 --T 12 --alpha 0.07
