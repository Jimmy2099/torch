# LLama example base on this framework - torch 

### llama3 1b cpu example

#### Run default example
docker run jimmy2099/torch:llama3_1b-CPU-7

#### Manual execution example
docker run --rm -it --entrypoint /bin/bash jimmy2099/torch:llama3_1b-CPU-7
./run.sh

##### System Requirements
- Ensure your system meets these requirements:
- Enough Memory to run example
- x86-64 CPU architecture
- Linux-based host system

##### Important Notes
1. Memory Requirements:
    - The Llama3 1B model in float32 precision requires significant memory resources
    - Insufficient memory may cause runtime errors:
      ```
      ./run.sh: line 34: 249 Killed    ./llama3_1b
      ```
2. Performance Considerations:
    - Expect longer inference times on CPU-only systems
    - GPU Example will be available in the future