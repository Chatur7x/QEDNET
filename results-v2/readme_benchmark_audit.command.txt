$ErrorActionPreference = 'Stop'
$readme = Get-Content README.md -Raw
$report = Get-Content results-v2/RESULTS_FINAL.md -Raw
$source = Get-Content results-v2/verify_parkinsons_resume.raw.txt -Raw
$checks = [ordered]@{
  benchmark_section = $readme.Contains('## Benchmark snapshot')
  report_link_exists = Test-Path 'results-v2/RESULTS_FINAL.md'
  report_mentions_four_datasets = $report.Contains('80-row stratified samples from each complete dataset') -or $report.Contains('80-row stratified samples from four biomedical datasets')
  exact_parkinsons_random_auc = $readme.Contains('0.8634920634920635') -and $source.Contains('mean=0.8634920634920635')
  exact_parkinsons_grouped_auc = $readme.Contains('0.7067460317460317') -and $source.Contains('mean=0.7067460317460317')
  leakage_caveat = $readme.Contains('leaked subject identity')
}
$checks.GetEnumerator() | ForEach-Object { '{0}={1}' -f $_.Key, $_.Value }
if ($checks.Values -contains $false) { exit 1 }
git diff --check
