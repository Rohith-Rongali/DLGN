for num_points in 40000; do
    for width in 200 500 1000; do
        export CUDA_VISIBLE_DEVICES=0
        python3 main.py data.dim_in=100 data.depth=4 data.num_points=$num_points train.num_epochs=700 model.width=$width
    done
done