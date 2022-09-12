# Setting up your ROS Workspace

## Part 0 - Install dependencies

In a terminal, run the following commands to install general system dependencies for Shutter's code:

```
sudo apt install python3-vcstool
```

> If you are setting up your workspace in the BIM laptops that are provided for the couse, then all apt dependencies have already been installed for you. If you encounter any problem about this, please contact the course T.F.

Then, install Python dependencies:

```
pip install --upgrade --user pip  # upgrade pip
pip install --user gdown          # install library to download Shutter simulation
```

When running commands on a terminal, pay attention to the information that is printed in the terminal. If you see any errors,
please post them in Slack and/or communicate with the course T.F.

## Part I - Set up your workspace to work with Shutter

*Catkin* is the official build system for ROS. To understand what it is for and why it exists, 
read sections 1, 2 and 4 of Catkin's conceptual overview document: 
[http://wiki.ros.org/catkin/conceptual_overview](http://wiki.ros.org/catkin/conceptual_overview).

Set up your Catkin workspace to work with the Shutter robot:

1. Create a [workspace](http://wiki.ros.org/catkin/workspaces) called *catkin_ws* 
in your home directory. Follow the steps in this tutorial: 
[http://wiki.ros.org/catkin/Tutorials/create_a_workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)

    > The [tutorial](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)
    page has tabs for switching between ROS distributions. Follow the tutorial for the distribution of
    ROS that you have installed in your system, i.e., Noetic.

2. Download Shutter's codebase into your workspace's `src` directory.

    ```bash
    # Go to the src folder in your workspace
    $ cd ~/catkin_ws/src

    # Clone the Shutter packages from GitLab
    $ git clone https://gitlab.com/interactive-machines/shutter/shutter-ros.git
 
    # Load git submodules with ROS dependencies
    $ cd shutter-ros
    $ git submodule update --init
    ```
    
    > [Git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) are other,
    external projects (Git repositories) that have been included in 
    the shutter-ros repository. These projects are needed to run the robot's base code.
    
    You should now have a number of directories in ~/catkin_ws/src/shutter-ros, including:
    
    ```bash
    $ cd ~/catkin_ws/src
    $ ls -C1 shutter-ros
    arbotix_ros
    documentation
    shutter_bringup
    shutter_description
    (...)
    ```
    
    Some of these directories are standard folders, other are ROS catkin packages. 
    A ROS catkin package contains:
    
    1. A [catkin compliant package.xml](http://wiki.ros.org/catkin/package.xml) file
    that contains basic information about the package, e.g., package name, description,
    license, author, dependencies, etc.
    
    2. A [CMakeLists.txt](http://wiki.ros.org/catkin/CMakeLists.txt) file that is 
    used by catkin to build the software package.
    
    For example, the shutter_bringup package has the following files:
    
    ```bash
    # Example
    $ ls -C1 ~/catkin_ws/src/shutter-ros/shutter_bringup
    CMakeLists.txt
    config
    launch
    package.xml
    README.md
    (...)
    ```
    
    > Each ROS package must have its own folder. This means that there cannot be
    nested packages. Multiple packages cannot share the same directory.
    
    Read the README.md file in the root level of the 
    [shutter-ros](https://gitlab.com/interactive-machines/shutter/shutter-ros.git) repository
    to understand its content and general organization. You can also access the documentation for shutter-ros at [https://shutter-ros.readthedocs.io](https://shutter-ros.readthedocs.io). 
        
4. Copy other custom packages (including [MoveIt](https://github.com/yale-img/moveit)) to your workspace:

    ```
    # clone dependencies
    $ cd ~/catkin_ws/src
    $ mkdir ros-planning
    $ vcs import --input shutter-ros/noetic_moveit.repos --recursive ros-planning
    ```

    Be patient as packages get downloaded into your `catkin_ws/src` folder.

5. Install other third-party dependencies with [rosdep](http://docs.ros.org/independent/api/rosdep/html/).
If rosdep is not found in your system, first install it and initialize it as 
indicated [here](http://docs.ros.org/independent/api/rosdep/html/overview.html). 
You will need sudo access to complete this step. 
 
    ```bash
    # update rosdep 
    $ rosdep update

    # install dependencies for Shutter
    $ cd ~/catkin_ws
    $ rosdep install -y -r --ignore-src --rosdistro=noetic --from-paths src
    ```

    > If you don't have pip installed, follow [these instructions](https://linuxconfig.org/how-to-install-pip-on-ubuntu-18-04-bionic-beaver) to install it before installing the Python dependencies for shutter_face.

    > If you are setting up your workspace in the BIM laptops that are provided for the couse, then all apt dependencies have already been installed for you. If you encounter any problem about this, please contact the course T.F.

           
6. Build the packages in the src directory of your workspace with `catkin_make`. 

    ```bash
    # Build your workspace
    $ cd ~/catkin_ws
    $ catkin_make -DCMAKE_BUILD_TYPE=Release
    ```

    Now you should have a devel space in `~/catkin_ws/devel`, which contains its own setup.bash file.
    Sourcing this file will `overlay` the install space onto your environment. 
    
    > Overlaying refers to building and using a ROS package from source on top of an existing version
    of that same package (e.g., installed at the system level in /opt/ros/noetic). For more information
    on overlaying, read [this tutorial](http://wiki.ros.org/catkin/Tutorials/workspace_overlaying).
    
    > Add ```source ~/catkin_ws/devel/setup.bash``` at the end of your `.bashrc` file
     to automatically set up your environment with your workspace every time you open a new shell.
     Otherwise, make sure to source ~/catkin_ws/devel/setup.bash on every new shell that you want
     to use to work with ROS. Sourcing setup.bash from your devel space will ensure that ROS 
     can work properly with the code that you've added to and built in ~/catkin_ws.
                  

At this point, you are done setting up your ROS workspace. If you want to know more about catkin workspaces, check out this [page](http://wiki.ros.org/catkin/workspaces). Importantly, note that the `catkin_make` command generated a `build` directory when it compiled the code in your `src` folder. This `build` directory has intermediary build files needed during the compilation process and you can delete it to recompile everything from scratch if you ever want to. Also, don't forget to source your `catkin_ws/devel/setup.sh` script whenever you are working with your ROS workspace. You can do this automatically by adding a line to your `~/.bashrc` file. For example, if your home folder is `/home/netid`, then add the following line to your `.bashrc`:

```
source /home/netid/catkin_ws/devel/setup.bash
```

If you don't use bash but Z shells, then source `setup.zsh` instead in your Z shell [startup file](https://zsh.sourceforge.io/Intro/intro_3.html).
