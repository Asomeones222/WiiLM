# Building WiiLM

---

## Prerequisites

* **DevKit Pro (devkitPPC + libogc)** installed and in your `PATH`.  You can grab it from [https://devkitpro.org](https://devkitpro.org).

---

## 1. Clone & Prepare

1. Open a shell on your dev machine:

   ```bash
   git clone https://github.com/Asomeones222/WiiLM.git
   cd WiiLM
   ```
2. Ensure `DEVKITPRO` and `DEVKITPPC` are set (usually in your shell profile) and update your PATH:

   ```bash
   export DEVKITPRO=/opt/devkitpro
   export DEVKITPPC=$DEVKITPRO/devkitPPC
   export PATH=$PATH:$DEVKITPRO/tools/bin:$DEVKITPPC/bin
   echo $DEVKITPPC  # should print something like /opt/devkitpro/devkitPPC
   ```

---

## 2. Build the App

From the project root, just run:

```bash
make
```

This produces:

* `WiiLM.elf`
* `WiiLM.dol`

The `.dol` is the Wii executable.


## Finally
The code is kinda messy but dw ur smart and will manage
Happy wii hacking!
