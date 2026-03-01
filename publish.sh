#!/bin/bash
# Build a single self-contained exe for Windows x64
dotnet publish src/NeuralSim.App/NeuralSim.App.csproj \
  -c Release \
  -r win-x64 \
  --self-contained true \
  -p:PublishSingleFile=true \
  -p:IncludeNativeLibrariesForSelfExtract=true

echo ""
echo "Output: src/NeuralSim.App/bin/Release/net9.0-windows/win-x64/publish/"
