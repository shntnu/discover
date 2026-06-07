{
  description = "TTT-Discover - pure uv dev shell (RL at test time for LLMs)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            # Most tasks run on 3.11; gpu_mode needs 3.13. uv picks the
            # interpreter per task via `uv venv --python 3.x`.
            python311
            python313
            uv
            git
            # System libraries that PyPI binary wheels (incl. torch) link against
            zlib
            stdenv.cc.cc.lib
          ];

          shellHook = ''
            echo "ttt-discover - pure uv dev shell"
            echo ""
            echo "Per-task isolated envs (frozen pins are the source of truth):"
            echo "  uv venv .venvs/math --python 3.11"
            echo "  uv pip install --python .venvs/math/bin/python -r requirements/requirements-math.txt"
            echo "  .venvs/math/bin/python -m examples.ac_inequalities.env"
            echo ""
            echo "  gpu_mode uses 3.13: uv venv .venvs/gpumode --python 3.13"
            echo ""
            unset PYTHONPATH

            # torch's CUDA wheels need the system NVIDIA driver. On NixOS it lives
            # at /run/opengl-driver/lib; on Ubuntu/macOS that path is absent and
            # the system linker already finds libcuda, so only add it when present.
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
          '';

          # Make nix-provided binary-wheel deps resolvable inside the shell.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.zlib
            pkgs.stdenv.cc.cc.lib
          ];
        };
      });
}
