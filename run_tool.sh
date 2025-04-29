ibstatus

nvidia-smi topo -m

set -x

echo "Running on $HOSTNAME"
echo "NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "AMLT_OUTPUT_DIR=$AMLT_OUTPUT_DIR"

# Set run variables
export RUN_N=8
export PPO_EPOCHS=4
export PROJECT_NAME="reasoning_tools"

# export DATASET_NAME="phi_math_tool"
export DATASET_NAME="phi_math_tool_subtasks"
# export MAX_LENGTH_CONTROL=65536
export MAX_RESPONSE_LENGTH=5120
export BASE_MODEL="phi-4"
# export BASE_MODEL="phi-4-o3-sft-4_1_25_long"
export LR=1e-7

# if $ARCH=="A100" then set the following
if [[ $ARCH == "A100" ]]; then
    export TENSOR_PARALLEL_SIZE=2
    export ULYSSES_PARALLEL_SIZE=1
    export PPO_BATCH_SIZE=$((NODES*8)) # This is batchsize of ppo
    export PPO_MAX_TOKEN_LENGTH=5120 # This is per GPU max token length
else
    export TENSOR_PARALLEL_SIZE=1
    export ULYSSES_PARALLEL_SIZE=1
    export PPO_BATCH_SIZE=$((5*NODES*8)) # This is batchsize of ppo
    export PPO_MAX_TOKEN_LENGTH=25600 # This is per GPU max token length
fi
export TRAIN_BATCH_SIZE=$((PPO_BATCH_SIZE)) # This is batchsize of the data loader
export SAVE_FREQ=50
export FP8_ADAM=true
export FP8_KVCACHE=true
export TOOL_USE_VLLM=true
export VLLM_ALLOCATION=0.7

# pip install -q vllm==0.8.1
# pip install -q tensordict==0.6.2
# pip install -q transformers==4.47.1

# if node rank is 0, start ray as head
if [ $NODE_RANK -eq 0 ]; then
    ray start --head --port=$MASTER_PORT
    sleep 60
else
    # wait for ray head to start
    sleep 10
    ray start --address=$MASTER_ADDR:$MASTER_PORT --block

    # finish with a sleep to keep the process alive
    echo "Ray should have started, sleeping..."
    sleep infinity
fi

# export RAY_ADDRESS="http://$MASTER_ADDR:$MASTER_PORT"

# check if ray is running on all nodes
ray status

# Run the script
chmod +x ./run_exp.sh
bash ./run_exp.sh