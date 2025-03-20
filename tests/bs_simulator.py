

batch_size = 96
sequence_length = 4736
device_batch_size = 2
ddp_world_size = 1
use_patch_level_training = True
patch_size = 4
val_tokens = 10002432


if use_patch_level_training:
    prev_device_batch_size = device_batch_size
    prev_train_accumulation_steps = batch_size // (prev_device_batch_size * ddp_world_size)
    device_batch_size = min(patch_size, prev_train_accumulation_steps) * prev_device_batch_size
    print(f"Using patch-level training. Modifying the device batch size to account for the patch size, from {prev_device_batch_size} to {device_batch_size}.")

# convenience variables
B, T = device_batch_size, sequence_length
# calculate the number of steps to take in the val loop.
assert val_tokens % (B * T * ddp_world_size) == 0
val_steps = val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = batch_size // (B * ddp_world_size)

print(f"Device batch size: {B}")
print(f"Total tokens batch size : {batch_size * T}")
print(f"Train accumulation steps: {train_accumulation_steps}")
print(f"Val steps: {val_steps}")
