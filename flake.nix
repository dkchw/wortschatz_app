{
  description = "German Vocabulary Frequency Analyzer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python3;
        pythonPackages = python.pkgs;

        # Helper to construct a spaCy model from its official release wheel
        buildSpacyModel = { pname, version, url, hash }:
          pythonPackages.buildPythonPackage {
            inherit pname version;
            format = "wheel";
            src = pkgs.fetchurl {
              inherit url hash;
            };
            propagatedBuildInputs = [ pythonPackages.spacy ];
          };

        de_core_news_sm = buildSpacyModel {
          pname = "de_core_news_sm";
          version = "3.8.0";
          url = "https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl";
          hash = "sha256-/saf7FKxeA8tJp1a91gqXigChzi9MZBTJFmutHO/o+c=";
        };

        de_core_news_md = buildSpacyModel {
          pname = "de_core_news_md";
          version = "3.8.0";
          url = "https://github.com/explosion/spacy-models/releases/download/de_core_news_md-3.8.0/de_core_news_md-3.8.0-py3-none-any.whl";
          hash = "sha256-uQP1kiDx523Wcqzap/pFTWcD/gVsXM1kV4IOcIdBFtA=";
        };

        de_core_news_lg = buildSpacyModel {
          pname = "de_core_news_lg";
          version = "3.8.0";
          url = "https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.8.0/de_core_news_lg-3.8.0-py3-none-any.whl";
          hash = "sha256-Nv2mUOR2tU1eh4A2NeNtrdHo4DTEtZYgiFhtaE9Mn+0=";
        };

        wortschatz = pythonPackages.buildPythonApplication rec {
          pname = "wortschatz";
          version = "2.2.0";
          format = "pyproject";

          src = ./.;

          postPatch = ''
            # Remove the PEP-508 URL specifiers from pyproject.toml to avoid offline build issues
            substituteInPlace pyproject.toml \
              --replace ' @ https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl' "" \
              --replace ' @ https://github.com/explosion/spacy-models/releases/download/de_core_news_md-3.8.0/de_core_news_md-3.8.0-py3-none-any.whl' "" \
              --replace ' @ https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.8.0/de_core_news_lg-3.8.0-py3-none-any.whl' ""
          '';

          nativeBuildInputs = [
            pythonPackages.setuptools
          ];

          propagatedBuildInputs = [
            pythonPackages.flask
            pythonPackages.spacy
            de_core_news_sm
            de_core_news_md
            de_core_news_lg
          ];

          doCheck = false;

          meta = with pkgs.lib; {
            description = "German Vocabulary Frequency Analyzer";
            homepage = "https://github.com/dkchw/wortschatz_app";
            license = licenses.asl20;
            mainProgram = "wortschatz";
          };
        };

      in
      {
        packages = {
          inherit wortschatz;
          default = wortschatz;
        };

        apps = {
          wortschatz = flake-utils.lib.mkApp { drv = wortschatz; };
          default = self.apps.${system}.wortschatz;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ wortschatz ];
        };
      }
    ) // {
      # System-agnostic Home Manager module
      homeManagerModules = {
        wortschatz = import ./nix/hm-module.nix self;
        default = self.homeManagerModules.wortschatz;
      };
    };
}
