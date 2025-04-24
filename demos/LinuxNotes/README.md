
<img width="1491" alt="Screenshot 2024-05-06 at 11 43 11 AM" src="https://github.com/adithya2424/EmbeddedEdgeAI/assets/34277400/30012935-43c4-4924-8500-14ef37645d1f">


### Yocto

Yocto is an open source project to create custom linux distributions regardless of the hardware architechture.

More info on Yocto is available at: https://www.yoctoproject.org

### Steps to create a custom linux image using Yocto for Beaglebone Black

Please follow the steps in below guide to create a customized image for BBB, we have meta layer which includes GStreamer framework.
[ https://medium.com/swlh/build-and-use-gstreamer-with-yocto-project-and-beaglebone-black-217d6822476d ]

Make sure to relax, make some memories and get back to this once you have built your image because the bitbake takes a lot of time depending on the system resources.

### Understanding the .wic files and the boot process 

We now have the .wic under "build/tmp/deploy/"

Use balena etcher tool to flash the .wic to the sdcard. Works like a charm.
[https://etcher.balena.io]

While the image is being flashed, please refer to the following figure to understand the Linux boot process.

<img width="1113" alt="Screenshot 2024-05-06 at 10 51 41 PM 1" src="https://github.com/adithya2424/EmbeddedEdgeAI/assets/34277400/8ce03b52-b05d-415a-a56c-9eb690862114">



