dotnet build -c Release
dotnet pack -c Release
Get-ChildItem bin/Release/*.nupkg | ForEach-Object { dotnet nuget push $_.FullName -k $env:NUGET_API_KEY -s https://api.nuget.org/v3/index.json --skip-duplicate }
Remove-Item -Path bin/Release/* -Recurse -Force