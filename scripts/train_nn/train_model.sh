#!/bin/bash -e
# This is a convenience wrapper that does a bit of extra heavy lifting for model training.

set -eo pipefail

source ./common_proc.sh
source ./config.sh

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "DERIVED_DATA_SUBDIR"
checkVarNonEmpty "SAMPLE_COLLECT_ARG"
checkVarNonEmpty "IR_MODELS_SUBDIR"

sampleProb=1
saveEpochSnapshots=0
saveEpochSnapshotsArg=""
noFinalValArg=""
amp=""

boolOpts=("h" "help" "print help"
          "amp" "amp" "use automatic mixed precision"
          "save_epoch_snapshots" "saveEpochSnapshots" "save snapshot after each epoch")

seed=0
epochQty=1
epochRepeatQty=1
batchesPerEpoch=0
masterPort=10001
deviceQty=1
batchSyncQty=4
deviceName="cpu"
addExperSubdir=""
modelConf=""
trainConf=""
initModelWeights=""
initModel=""
optim="adamw"
momentum="0.9"
validCheckPoints=""
validRunDir=""
maxQueryVal=""
valType=""
batchesPerEpoch=""
distrBackend="gloo"

paramOpts=("seed"          "seed"             "seed (default $seed)"
      "optim"              "optim"            "optimizer (default $optim)"
      "momentum"           "momentum"         "SGD momentum (default $momentum)"
      "epoch_qty"          "epochQty"         "# of epochs (default $epochQty)"
      "epoch_repeat_qty"   "epochRepeatQty"   "# of epoch epoch repetition (default $epochRepeatQty)"
      "batches_per_train_epoch"  "batchesPerEpoch"  "# of batches per train epoch (default $batchesPerEpoch)"
      "max_query_val"      "maxQueryVal"      "max # of val queries"
      "valid_checkpoints"  "validCheckPoints" "validation checkpoints (in # of batches)"
      "valid_run_dir"      "validRunDir"      "directory to store full predictions on validation set"
      "master_port"        "masterPort"       "master port for multi-GPU train (default $masterPort)"
      "device_name"        "deviceName"       "device name for single-gpu train (default $deviceName)"
      "device_qty"         "deviceQty"        "# of device (default $deviceQty)"
      "batch_sync_qty"     "batchSyncQty"     "# of batches before model sync"
      "add_exper_subdir"   "addExperSubdir"   "additional experimental sub-directory (optional)"
      "model_conf"         "modelConf"        "model JSON configuration file"
      "train_conf"         "trainConf"        "training JSON configuration file"
      "init_model_weights" "initModelWeights" "initial model weights"
      "init_model"         "initModel"        "init model"
      "valid_type"         "valType"          "validation type: always (every epoch), last_epoch (last epoch), never"
      "distr_backend"      "distrBackend"     "Pytorch backend for distributed processing"
)

parseArguments $@

usageMain="<collection> <train data subdir (relative to derived data)> <model type>"

if [ "$help" = "1" ] ; then
  genUsage "$usageMain"
  exit 1
fi

collect=${posArgs[0]}

if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit 1
fi

derivedDataDir="$COLLECT_ROOT/$collect/$DERIVED_DATA_SUBDIR"

trainSubDir=${posArgs[1]}

if [ "$trainSubDir" = "" ] ; then
  genUsage "$usageMain" "Specify training data subdir relative to $derivedDataDir (2d arg)"
  exit 1
fi

modelName=${posArgs[2]}
if [ "$modelName" = "" ] ; then
  genUsage "$usageMain" "Specify model name/type, e.g., vanilla_bert (3rd arg)"
  exit 1
fi

initModelArgs=""
if [ "$initModelWeights" != "" ] ; then
  initModelArgs=" --model_name $modelName --init_model_weights $initModelWeights "
elif [ "$initModel" != "" ] ; then
  initModelArgs=" --init_model $initModel "
else
  initModelArgs=" --model_name $modelName "
  echo "WARNING: neither -init_model_weights nor -init_model specified, training from random init!"
fi

outModelDir="$derivedDataDir/$IR_MODELS_SUBDIR/$modelName/$addExperSubdir/$seed/"
trainDir="$derivedDataDir/$trainSubDir"

# Copy both JSON configurations to the output directory
if [ "$modelConf" != "" ] ; then
  cp "$COLLECT_ROOT/$collect/$modelConf" "$outModelDir"
  bn=`basename "$modelConf"`
  modelConfDest="$outModelDir/$bn"
  modelConfArg=" --json_model_conf $modelConfDest "
  echo "Model JSON config:                            $modelConfDest"
fi

if [ "$trainConf" != "" ] ; then
  cp "$COLLECT_ROOT/$collect/$trainConf" "$outModelDir"
  bn=`basename "$trainConf"`
  trainConfDest="$outModelDir/$bn"
  trainConfArg=" --json_train_conf $trainConfDest "
  echo "Training JSON config:                         $trainConfDest"
fi

echo "=========================================================================="

python -u ./train_nn/train_model.py \
  $initModelArgs \
  $ampArg \
  $modelConfArg \
  $trainConfArg \
  $validCheckPointsArg \
  $validRunDirArg \
  $maxQueryValArg \
  $valTypeArg \
  $batchesPerEpochArg \
  --optim $optim \
  --momentum $momentum \
  --seed $seed \
  --device_name $deviceName \
  --device_qty $deviceQty \
  --distr_backend $distrBackend \
  --batch_sync_qty $batchSyncQty \
  --epoch_qty $epochQty \
  --epoch_repeat_qty $epochRepeatQty \
  $saveEpochSnapshotsArg \
  --master_port $masterPort \
  --datafiles "$trainDir/data_query.tsv"  \
              "$trainDir/data_docs.tsv" \
   --train_pairs "$trainDir/train_pairs.tsv" \
  --valid_run "$trainDir/test_run.txt" \
  --qrels "$trainDir/qrels.txt" \
  --model_out_dir "$outModelDir" \
2>&1|tee "$outModelDir/train.log"
