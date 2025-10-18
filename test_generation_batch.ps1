# Batch generation test with optimal settings
# Based on analysis: big size with CFG 5.0 works best

$checkpoint = "models/last.ckpt"
$cfg = 5.0
$size = "big"

# Test different tree types
$prompts = @(
    "oak_tree",
    "birch_tree", 
    "spruce_tree",
    "jungle_tree",
    "pine_tree",
    "snowy_pine"
)

Write-Host "="*60
Write-Host "Batch Generation Test - Optimal Settings"
Write-Host "Size: $size | CFG: $cfg"
Write-Host "="*60

foreach ($prompt in $prompts) {
    Write-Host "`nGenerating: $prompt"
    python scripts/generate_discrete_diffusion.py `
        --checkpoint $checkpoint `
        --prompt $prompt `
        --size $size `
        --guidance-scale $cfg `
        --output "generated/optimal_test"
}

Write-Host "`n"
Write-Host "="*60
Write-Host "Generation complete! Check: generated/optimal_test/"
Write-Host "="*60
