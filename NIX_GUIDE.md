# Wortschatz – Nix & Home Manager Installation Guide

This guide describes how to install and manage **Wortschatz** using **Nix Flakes** and **Home Manager**.

With the provided Nix configuration, you can:
- Install the `wortschatz` CLI tool directly to your user profile.
- Run Wortschatz as a systemd user daemon, automatically starting it when you log in and keeping it running in the background.

---

## ⚡ Prerequisites

Ensure you have **Nix** installed with **Flakes** enabled. If Flakes are not yet enabled, add the following to your `/etc/nix/nix.conf` or `~/.config/nix/nix.conf`:

```conf
experimental-features = nix-command flakes
```

---

## 📦 Option 1: Standalone Home Manager Configuration

If you manage your Home Manager configuration standalone (e.g., via `home-manager switch`), update your files as follows:

### 1. Update `flake.nix` of your home-manager configuration

Add the Wortschatz repository to your `inputs` and pass it to your outputs:

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Add the Wortschatz input pointing to the repository
    wortschatz = {
      url = "git+file:///run/host/home/dkchw/Documents/Code/Ongoing/Repo/wortschatz_app"; # Or Github URL if hosted online
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, home-manager, wortschatz, ... }: {
    homeConfigurations."yourusername" = home-manager.lib.homeManagerConfiguration {
      pkgs = import nixpkgs { system = "x86_64-linux"; }; # Adjust system accordingly
      modules = [
        ./home.nix
        # Import the Wortschatz Home Manager module
        wortschatz.homeManagerModules.default
      ];
    };
  };
}
```

### 2. Configure `home.nix`

Now you can enable either the CLI tool or the background web daemon inside your `home.nix` configuration:

```nix
{ config, pkgs, ... }: {
  # 1. Enable the CLI command 'wortschatz'
  programs.wortschatz.enable = true;

  # 2. (Optional) Enable the background local web service
  services.wortschatz = {
    enable = true;
    port = 5000; # The port where the local web server runs (default: 5000)
    preloadPaths = [
      "/home/yourusername/Documents/vocabulary-lists/" # Folders/files to preload on startup
    ];
  };
}
```

Apply your changes:
```bash
home-manager switch --flake .#yourusername
```

---

## ❄️ Option 2: NixOS + Home Manager Integration

If you import Home Manager as a module within your NixOS system configuration:

### 1. Update your NixOS `/etc/nixos/flake.nix`

Add the input and pass it down to your modules:

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    wortschatz = {
      url = "git+file:///run/host/home/dkchw/Documents/Code/Ongoing/Repo/wortschatz_app";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, home-manager, wortschatz, ... }@inputs: {
    nixosConfigurations."yourhostname" = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      specialArgs = { inherit inputs; };
      modules = [
        ./configuration.nix
        home-manager.nixosModules.home-manager
        {
          home-manager.useGlobalPkgs = true;
          home-manager.useUserPackages = true;
          home-manager.users.yourusername = {
            imports = [
              # Import the module
              wortschatz.homeManagerModules.default
            ];
          };
        }
      ];
    };
  };
}
```

### 2. Configure in your Home Manager block

Add the configuration options inside your user's home manager config block:

```nix
{ inputs, ... }: {
  # ...
  programs.wortschatz.enable = true;
  services.wortschatz = {
    enable = true;
    port = 5000;
  };
}
```

Rebuild your system:
```bash
sudo nixos-rebuild switch --flake .#yourhostname
```

---

## 🛠️ Usage & Operations

### CLI Tool
Once installed, the CLI tool is available in your `$PATH`:
```bash
# Start server manually
wortschatz serve

# Preload files and run on custom port
wortschatz serve ~/text.txt --port 8080
```

### Systemd User Daemon Service
If you enabled `services.wortschatz.enable = true`, Home Manager registers a systemd user unit. You can control it using the standard user systemctl commands:

```bash
# Check the status of the service
systemctl --user status wortschatz

# Restart the service (useful if you want to clear the in-memory uploaded session files)
systemctl --user restart wortschatz

# Stop the daemon
systemctl --user stop wortschatz

# View real-time service logs
journalctl --user -u wortschatz -f
```

---

## 🧪 Ad-hoc Running / Testing (No Installation)

You can run Wortschatz directly from the Flake without installing it onto your system:

```bash
# Run the default app
nix run git+file:///run/host/home/dkchw/Documents/Code/Ongoing/Repo/wortschatz_app -- serve

# Run in a temporary shell with the command available
nix shell git+file:///run/host/home/dkchw/Documents/Code/Ongoing/Repo/wortschatz_app -c wortschatz serve
```
