# GPT-5 Codex Audit

## Scope
- Repository: `m4-training`
- Commit target: `main`
- Focus files: `train.py`, datasets loaders, training guides

## Key Findings

1. **WeightedTrainer not invoked**  
   - Reference: `train.py`, `WeightedTrainer` class definition (lines ~24-48) and trainer instantiation (lines ~247-269).  
   - Impact: Training previously defaulted to `Trainer`, so class imbalance handling was skipped despite documentation.  
   - Resolution: We now compute class weights from the stratified training split (`train.py`, lines ~176-200) and instantiate `WeightedTrainer` with `class_weights` when `--use_class_weights` is passed (lines ~248-269).

2. **Validation generator default missing in cleaned English set**  
   - Reference: `train.py`, argument parser (lines ~51-74).  
   - Issue: Default `--val_generator gpt2-neo` pointed to a non-existent cleaned file, yielding an empty validation set.  
   - Fix: Default updated to `flant5`, which exists in `M4_cleaned/data`, and non-stratified flows now raise a clear error if the generator is absent (lines ~137-162).

3. **Reddit data excluded from actual pipeline**  
   - Reference: `train.py`, data loading block (lines ~121-170).  
   - Problem: `--reddit_file` flag was parsed but never used, so the 60k human set never reached training.  
   - Remedy: Added `load_reddit_data` helper (lines ~96-119) and merged those rows with the M4 corpus before splitting.

4. **Stratified split missing despite CLI flag**  
   - Reference: `train.py`, stratified branch (lines ~137-169).  
   - Observation: The previous `--stratified_split` flag was unused; we now honor it by performing a stratified 90/10 split (configurable via `--validation_split`) across the combined M4 + Reddit pool using `train_test_split`.  
   - Outcome: Validation now reflects the class imbalance and domain mixture seen in training.

5. **Reddit samples not chunked**  
   - Reference: `train.py`, `create_chunked_dataset` usage (lines ~202-233).  
   - Concern: Reddit posts bypassed the chunk/overlap logic, leading to hard truncation at 512 tokens.  
   - Change: Both training and validation datasets—regardless of origin—are passed through `create_chunked_dataset`, guaranteeing consistent 512-token windows with overlap.

## Open Considerations
- Review guidance in `TRAINING_GUIDE_COMPLETE.md` to reflect the updated CLI defaults (`--use_class_weights`, generator list, stratified behaviour).  
- Consider exposing a CLI knob for Reddit validation sampling if you ever need generator-based holdout while still covering casual text.

## Next Steps
1. Run a dry run with `--max_train_samples`/`--max_val_samples` on local hardware to confirm label ratios and chunk counts.  
2. Sync documentation to highlight the new `--validation_split` option and default generator list.  
3. Prep RunPod scripts (`example_commands.sh`, `setup_runpod.sh`) to pass `--use_class_weights --stratified_split` explicitly.
