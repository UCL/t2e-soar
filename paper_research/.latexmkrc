# Enable PNG support
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode';

# Add rules for PNG to PDF conversion
$clean_ext = 'synctex.gz synctex.gz(busy) run.xml tex.bak bbl bcf fdb_latexmk run tdo %R-blx.bib';
