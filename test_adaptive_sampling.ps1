# Test Adaptive Sampling Improvements
# Compare old vs new sampling methods

$checkpoint = "models/last.ckpt"
$output_dir = "generated/adaptive_test"

Write-Host "="*60
Write-Host "Testing Adaptive Sampling Improvements"
Write-Host "="*60
Write-Host ""

# Test 1: Default settings (auto-adaptive)
Write-Host "1. Testing auto-adaptive settings..."
python scripts/generate_discrete_diffusion.py `
    --checkpoint $checkpoint `
    --prompt "oak_tree" `
    --size big `
    --guidance-scale 5.0 `
    --output "$output_dir/auto"

Write-Host ""

# Test 2: Conservative sampling (less chaos)
Write-Host "2. Testing conservative sampling..."
python scripts/generate_discrete_diffusion.py `
    --checkpoint $checkpoint `
    --prompt "oak_tree" `
    --size big `
    --guidance-scale 5.0 `
    --temperature 0.8 `
    --top-k 10 `
    --top-p 0.85 `
    --output "$output_dir/conservative"

Write-Host ""

# Test 3: Huge size with adaptive schedule
Write-Host "3. Testing huge size with adaptive noise..."
python scripts/generate_discrete_diffusion.py `
    --checkpoint $checkpoint `
    --prompt "oak_tree" `
    --size huge `
    --guidance-scale 5.0 `
    --temperature 0.8 `
    --output "$output_dir/huge_adaptive"

Write-Host ""

# Test 4: Different tree types to test prompt following
Write-Host "4. Testing different tree types..."
$trees = @("oak_tree", "birch_tree", "spruce_tree")
foreach ($tree in $trees) {
    Write-Host "  - Generating $tree..."
    python scripts/generate_discrete_diffusion.py `
        --checkpoint $checkpoint `
        --prompt $tree `
        --size big `
        --guidance-scale 5.0 `
        --temperature 0.8 `
        --top-k 12 `
        --output "$output_dir/trees"
}

Write-Host ""
Write-Host "="*60
Write-Host "Testing complete! Check: $output_dir"
Write-Host ""
Write-Host "Expected improvements:"
Write-Host "  - Fewer stone/gray blocks (top-k filtering)"
Write-Host "  - Better structure in huge size (adaptive noise)"
Write-Host "  - More coherent trees (top-p filtering)"
Write-Host "="*60
