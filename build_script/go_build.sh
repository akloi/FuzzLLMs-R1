cd ../target/go/
if [ ! -d "go" ]; then
    if [ ! -f "go1.24.6.linux-amd64.tar.gz" ]; then
        wget https://go.dev/dl/go1.24.6.linux-amd64.tar.gz
    fi
    tar -xzf go1.24.6.linux-amd64.tar.gz
fi

CURRENT_DIR=$(pwd)
export PATH=$PATH:$CURRENT_DIR/go/bin
export GOROOT_BOOTSTRAP=$CURRENT_DIR/go
git clone https://go.googlesource.com/go goroot
cd goroot
cd src
./make.bash