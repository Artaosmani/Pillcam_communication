#!/bin/sh
# Re-create every number in results/RESULTS.md from the trained models (run from Anomally_detection/).
#   sh run_all.sh          # evaluate + export + benchmark + pipeline + report (~30 min, uses existing weights)
#   sh run_all.sh train    # train both classifiers and YOLO26n / P2 first (~10 h on the M3)
# Close other heavy programs first: bench.py and pipeline.py measure latency.
set -e
cd "$(dirname "$0")"

if [ "$1" = "train" ]; then
  python Classifier/train.py --name classifier_96
  python Classifier/train.py --set imgsz=84 --name classifier_84
  python YOLO/train.py --name yolo26n
  python YOLO/train.py --model Configs/yolo26.yml --name yolo26n_p2
fi

python YOLO/test_boxes.py
python ../Board/test_receiver.py
for run in yolo26n yolo26n_p2; do
  w=runs/$run/weights/best.pt
  [ -f "$w" ] || continue
  python YOLO/export.py   --weights "$w"
  python YOLO/evaluate.py --weights "$w"
  python YOLO/bench.py    --weights "$w"
done
python pipeline.py
python report.py
