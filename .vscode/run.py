import shlex

def generate_launch_arguments(input_string):
    # Split the input string into individual arguments
    args = shlex.split(input_string)

    # Remove any leading or trailing whitespace from each argument
    args = [arg.strip() for arg in args]

    return args

# Example usage
input_string = 'IMLE --suffix exp_1_1 --save_folder /home/samp8/scratch/Mini3MRL/ --gpus 0 --data_tr mnist --data_val mnist --eval_iter 50 --save_iter 50 --probe_linear 1 --time 03:00:00 --env pip --env_dir /home/samp8/PythonENVs/py3103MRL/ --mem 30'
output_list = generate_launch_arguments(input_string)
output_list = str(output_list).replace("'", '"')
print(output_list)