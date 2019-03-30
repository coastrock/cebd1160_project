To build in Ubuntu18.04:

The container can be built using the following command from within this directory:

sudo docker build -t coastrock/cebd1160-project .

To run in Ubuntu 18.04:

The software can be run from within this directory with the following command:

sudo docker run -ti -v ${PWD}:${PWD} coastrock/cebd1160-project 

