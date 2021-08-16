---
title: Configure OpenCV for Xcode
date: 2020-10-13
tags:
  - OpenCV
  - IDE
  - macOS
---

Xcode is a complete developer toolset for creating apps for Mac, iPhone, iPad, Apple Watch, and Apple TV. Xcode brings user interface design, coding, testing, debugging, and submitting to the App Store all into a unified workflow.

Here's a tutorial to setting up OpenCV and C++ environment for Xcode on macOS.

<!-- more -->

## Pre Requirements

- Install Homebrew

Aptly titled _"The missing package manager for macOS"_ is enough said. Homebrew is the macOS equivalent of the Ubuntu/Debian-based `apt-get`.

```bash
/usr/bin/ruby -e $(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)
```

- Install OpenCV

Simply we use _brew_ to install the latest pre-compiled _OpenCV_ to our computer:

```bash
brew install opencv
```

- Install pkg-config

The pkg-config is a helper tool used when compiling applications and libraries. It helps you insert the correct compiler options on the command line rather than hard-coding values. This will be helpful for finding the correct linker flags for OpenCV. This will be more clear in the subsequent steps.

```bash
brew install pkg-config
```

- View OpenCV linker flags

To view the linker flags for OpenCV, we should find `.pc` file first. It's common to see this file in:

```plain
/usr/local/Cellar/opencv/<version_number>/lib/pkgconfig/opencv.pc
```

Then view the linker flags by specifing the location of opencv.pc file:

```bash
pkg-config --cflags --libs <path/to/opencv.pc>
```

## Commandline

Running code in the terminal, the code should be compiled first:

```bash
g++ $(pkg-config --cflags --libs <path/to/opencv.pc>) -std=c++17  main.cpp -o main.o
```

Then run the binary:

```bash
chmod +x main.o
./main.o
```

## Xcode developing

### Configure enviroment

- Create project

  Before following the below steps to run OpenCV C++ code in Xcode, a C++ project in Xcode should be created first.

  1. Click on `File > New > Project`
  1. Under Choose a template for new project click on **macOS**
  1. Under Application click on **Command Line Tool**
  1. Get the above screen. Fill in the details and set the Language to **C++**.

- Set Header Search Paths

  1. To set Header Search Path in Xcode, first click on the Xcode project, then go to **Build Settings** and then search for **Header Search Paths**.
  1. Set the Header Search Path to the path of OpenCV include folder:

```plain
/usr/local/Cellar/opencv/<version_number>/include/opencv4/opencv2
/usr/local/Cellar/opencv/<version_number>/include/opencv4
/usr/local/Cellar/opencv/<version_number>/include
```

- Set Library Search Paths

  1. In this case, follow steps similar to _Set Header Search Paths_ above but search for **Library Search Paths** in the search bar.
  1. Set the Library Search Path to the path of OpenCV library folder.

```plain
/usr/local/Cellar/opencv/<version_number>/lib
```

- Set Other Linker Flags

  1. Search for **Other Linker Flags** in the search bar.
  1. Set the other linker flags with all the flag values obtained after running `pkg-config` command above.

### Apply permitions

- To allow camera access, first click on the Xcode project, then go to **TARGETS > Signing & Capabilities** and check **Disable Library Validation**, **Debugging Tool**, **Audio Input** and **Camera**.

- Create `Info.plist`, add access requirement details:

```plist
Key: Privacy - Camera Usage Description
Value: $(PRODUCT_NAME) camera use
```

  Then go to **TARGETS > Build Phases > Copy Files**, select _Destnation_ to **Products Directory**, leave _Subpath_ blank and click `+` to select file `Info.plist`.
