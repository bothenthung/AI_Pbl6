import sys
sys.path.append("..")
import utils.diff_match_patch as dmp_module
import re

def diff_wordMode(text1, text2):
  dmp = dmp_module.diff_match_patch()
  a = dmp.diff_linesToWords(text1, text2)
  lineText1 = a[0]
  lineText2 = a[1]
  lineArray = a[2]
  diffs = dmp.diff_main(lineText1, lineText2, False)
  dmp.diff_charsToLines(diffs, lineArray)
  return diffs

def correctFunction(text, corrector):
  out = corrector.correct_transfomer_with_tr(text, num_beams=1)
  return out


def postprocessing_result(out):
  noised_text = out['original_text']
  predict_text = out['predict_text']
  print(noised_text, file=sys.stderr)
  print(predict_text, file=sys.stderr)
  diff = diff_wordMode(noised_text, predict_text)
  result = []
  for i, entry in enumerate(diff):
    if entry[0] == 0:
      result.append(entry)
    elif entry[0] == -1:
      if i + 1 < len(diff) and diff[i + 1][0] == 1:
        result.append((1, entry[1], diff[i + 1][1]))
      else:
        result.append((1, entry[1], " ") )
    else:
      if i - 1 >= 0 and diff[i - 1][0] == -1:
        continue
      else:
        result.append((1, " ", entry[1]) )
  print(result, file=sys.stderr)
  return result

