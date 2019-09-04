# CS486 - Artificial Intelligence
[![Binder](http://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/usma-eecs/cs486/master?urlpath=lab)

This repository uses submodules, to clone use:

```bash
$ git clone --recurse-submodules https://github.com/usma-eecs/cs486.git 
```

When updating, make sure your pull includes submodules:

```bash
$ git pull --recurse-submodules
```

If you forget to do that (or are reading this too late) you can fix it by running the following commands from the repo's root directory:

```bash
$ git submodule init
$ git submodule update
```

The in-class exercises make use of Jupyter. Recommend installing [Anaconda 3](https://www.anaconda.com/download/#windows), which includes Jupyter and JupyterLab. **It is not recommended to install Anaconda for "All Users", otherwise you will have to run the Anaconda Prompt as an Administrator!**

If you do not have `git` on your workstation, then you can install it using Anaconda:

* Open the *Anaconda Prompt*
* Run the command: *conda install git* (You may need to disable McAffee's Quick Scan on Army machines). 
* You can now run the *git clone* command above in the *Anaconda Prompt*.

The AIMA visualization are available on the [aima-javascript page](http://aimacode.github.io/aima-javascript/).
