self: { config, lib, pkgs, ... }:

with lib;

let
  cfgProg = config.programs.wortschatz;
  cfgServ = config.services.wortschatz;
  # Reference the default package build from the flake for the current system
  wortschatz-pkg = self.packages.${pkgs.system}.default;
in
{
  options = {
    programs.wortschatz = {
      enable = mkEnableOption "Wortschatz (German Vocabulary Frequency Analyzer) CLI";
      package = mkOption {
        type = types.package;
        default = wortschatz-pkg;
        description = "The Wortschatz package to install.";
      };
    };

    services.wortschatz = {
      enable = mkEnableOption "Wortschatz local web server daemon";
      package = mkOption {
        type = types.package;
        default = wortschatz-pkg;
        description = "The Wortschatz package to use for the systemd service.";
      };
      port = mkOption {
        type = types.port;
        default = 5000;
        description = "Port for the local web server.";
      };
      preloadPaths = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "File or folder paths to preload into the Wortschatz session.";
      };
    };
  };

  config = mkMerge [
    # Install the CLI tool
    (mkIf cfgProg.enable {
      home.packages = [ cfgProg.package ];
    })

    # Set up the background systemd service daemon
    (mkIf cfgServ.enable {
      home.packages = [ cfgServ.package ];

      systemd.user.services.wortschatz = {
        Unit = {
          Description = "Wortschatz - German Vocabulary Frequency Analyzer Daemon";
          After = [ "network.target" ];
        };
        Service = {
          # Wortschatz serve runs the Flask server on the specified port preloading any paths
          ExecStart = "${cfgServ.package}/bin/wortschatz serve --port ${toString cfgServ.port} ${escapeShellArgs cfgServ.preloadPaths}";
          Restart = "on-failure";
        };
        Install = {
          WantedBy = [ "default.target" ];
        };
      };
    })
  ];
}
