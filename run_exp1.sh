for dim_in in 30 40 50; do
    for width in 200 500; do
        num_epochs=$((250 + 6 * $dim_in))
        python3 main.py data.dim_in=$dim_in data.depth=4 data.num_points=50000 train.num_epochs=$num_epochs model.width=$width  
    done
done