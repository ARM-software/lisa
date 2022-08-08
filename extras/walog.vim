" Copy this into ~/.vim/syntax/ and add the following to your ~/.vimrc:
"     au BufRead,BufNewFile run.log set filetype=walog
"
if exists("b:current_syntax")
  finish
endif

syn region debugPreamble start='\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d DEBUG' end=':' 
syn region infoPreamble start='\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d INFO' end=':' 
syn region warningPreamble start='\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d WARNING' end=':' 
syn region errorPreamble start='\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d ERROR' end=':' 
syn region critPreamble start='\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d CRITICAL' end=':' 

hi debugPreamble guifg=Blue  ctermfg=DarkBlue
hi infoPreamble guifg=Green  ctermfg=DarkGreen
hi warningPreamble guifg=Yellow  ctermfg=178
hi errorPreamble guifg=Red  ctermfg=DarkRed
hi critPreamble guifg=Red  ctermfg=DarkRed cterm=bold gui=bold

let b:current_syntax='walog'

