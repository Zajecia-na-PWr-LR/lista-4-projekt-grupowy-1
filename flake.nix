{
  inputs = {
    nixpkgs.url = "nixpkgs";
  };

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      
      pkgs = import nixpkgs {
        inherit system;
        overlays =  (import /etc/nixos/overlays.nix {}).nixpkgs.overlays;
      };

    in {
      devShells.${system}.default = pkgs.mkJupyterShell {
        packages = ps: with ps; [
          pandas
          pandas-stubs
          seaborn
          matplotlib
          numpy
          scikit-learn
        ];
      };
    };
}
