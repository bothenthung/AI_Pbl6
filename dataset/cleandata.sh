#!/bin/bash
corpus=$1
root=../data/$1/
echo "Clean corpus $1"
cat $root$corpus.train[0-9]* > $root$corpus.train
rm -r $root$corpus.train[0-9]*
cat $root$corpus.test[0-9]* > $root$corpus.test
rm -r $root$corpus.test[0-9]*
cat $root$corpus.train.noise[0-9]* > $root$corpus.train.noise
rm -r $root$corpus.train.noise[0-9]*
cat $root$corpus.test.noise[0-9]* > $root$corpus.test.noise
rm -r $root$corpus.test.noise[0-9]*
cat $root$corpus.length.train[0-9]* > $root$corpus.length.train
rm -r $root$corpus.length.train[0-9]*
cat $root$corpus.length.test[0-9]* > $root$corpus.length.test
rm -r $root$corpus.length.test[0-9]*
cat $root$corpus.valid.noise[0-9]* > $root$corpus.valid.noise
rm -r $root$corpus.valid.noise[0-9]*
cat $root$corpus.length.valid[0-9]* > $root$corpus.length.valid
rm -r $root$corpus.length.valid[0-9]*
cat $root$corpus.valid[0-9]* > $root$corpus.valid
rm -r $root$corpus.valid[0-9]*
cat $root$corpus.onehot.test[0-9]* > $root$corpus.onehot.test
rm -r $root$corpus.onehot.test[0-9]*
cat $root$corpus.onehot.train[0-9]* > $root$corpus.onehot.train
rm -r $root$corpus.onehot.train[0-9]*
cat $root$corpus.onehot.valid[0-9]* > $root$corpus.onehot.valid
rm -r $root$corpus.onehot.valid[0-9]*