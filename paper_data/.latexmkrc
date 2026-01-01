# Configuration for latexmk
# Use pdflatex engine for proper PNG image support
$pdf_mode = 1;
$postscript_mode = 0;
$dvi_mode = 0;

# Enable SyncTeX for editor integration
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode %O %S';

# Auto-clean auxiliary files after build
$clean_ext = 'auxlock fls';

# Enable continuous watch mode (optional)
# $preview_mode = 1;
