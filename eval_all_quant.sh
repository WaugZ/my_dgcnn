#!/usr/local/bin/bash

#$1 is dir $2...$n are script params
if [ $# -lt 1 ]; then
    echo "\$1 is dir_name the others are eval params, remember type with --"
    exit 0
fi

params=""
for ((i=2;i<=$#;i++)); do
    params="$params ${!i}"
done
# echo $params

base_dir=$1
prefix="model"
backfix=.meta

if [ -f "$base_dir/model-r-5.ckpt.meta" ]; then
    prefix="model-r"
fi

best_score=0.0
best_ind=0
awk_re="/eval|airplane:|bathtub:|bed:|bench:|bookshelf:|"
awk_re=$awk_re"bottle:|bowl:|car:|chair:|cone:|cup:|curtain:|desk:|door:|"
awk_re=$awk_re"dresser:|flower_pot:|glass_box:|guitar:|keyboard:|lamp:|laptop:|"
awk_re=$awk_re"mantel:|monitor:|night_stand:|person:|piano:|plant:|radio:|"
awk_re=$awk_re"range_hood:|sink:|sofa:|stairs:|stool:|table:|tent:|toilet:|"
awk_re=$awk_re"tv_stand:|vase:|wardrobe:|xbox:/"

for ((i=0;i<=250;i+=5)); do
    echo "now $i current best score is $best_score in $prefix-$best_ind"
    model="$prefix-$i.ckpt"
    full_name="$model$backfix"
    if [ -f "$base_dir/$full_name" ]; then
        python evaluate.py --model_path=$base_dir/$model $params > "$base_dir/tmp_log"
        score=`awk '/eval accuracy:/{print $3}' "$base_dir/tmp_log"`
        # echo $score
        if [ `echo "$best_score < $score"|bc` -eq 1 ]; then
            # echo "haha"
            best_score=$score
            best_ind=$i
            echo "$prefix-$i.ckpt" > "$base_dir/bestNow"
            awk $awk_re "$base_dir/tmp_log" >> "$base_dir/bestNow"
            # cat "$base_dir/tmp_log" | grep "." >> "$base_dir/bestNow"
        fi
    else
        echo "pass"
    fi
done

echo "best score is $best_score in $best_ind"

if [ -f "$base_dir/tmp_log" ]; then
    rm "$base_dir/tmp_log"
fi
# python evaluate.py --quanti=1 --concat=0 --dyna=1 --scale=.5 --model_path=$1