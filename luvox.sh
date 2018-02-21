#!/usr/bin/env bash

echo "Hello World! This is LuVoX talking!"
echo "These are my parameters: $0 $@"

echo "$(date) : $0 $@" >> luvox.log
LUVOX_PREFIX="/tmp/LuVoX/$(date +%s)"
mkdir -p "$LUVOX_PREFIX"
echo "Created prefix path: $LUVOX_PREFIX"

infolder="$1"

./luvox.py --verbose --debug --prefix "$LUVOX_PREFIX" $@ &> "${LUVOX_PREFIX}.log"

for d in $(find "${LUVOX_PREFIX}/" -maxdepth 1 -mindepth 1 -type d); do
    echo "Processing study $(basename ${d})"

    mergedpdf="${d}/$(basename ${d}).pdf"
    # delete study pdf if present
    if [ -f ${mergedpdf} ]; then
        rm ${mergedpdf}
    fi
    pdfcdcm=${mergedpdf/%.pdf/.dcm}
    echo "Ouput PDF DCM: ${pdfcdcm}"

    # merge all pdfs
    pdfs=$(find "${d}" -type f -iname "*.pdf")
    gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=${mergedpdf} ${pdfs}

    # convert pdf2dcm
    # one of the input files as scaffold
    dcmscaffold=$(find "${infolder}" -type f -print -quit)
    echo "dcmscaffold: ${dcmscaffold}"
    pdf2dcm +st ${dcmscaffold} ${mergedpdf} ${pdfcdcm}

    # set StudyDescription
    StudyDescription=$(dcmdump +P "0008,1030" ${dcmscaffold} | grep -E -o "\[.*\]")
    StudyDescription=${StudyDescription:1:${#StudyDescription}-2}
    dcmodify -i "(0008,1030)=${StudyDescription}" ${pdfcdcm}
    echo "Setting StudyDescription: ${StudyDescription}"

    # set ReferringPhysicianName
    dcmodify -i "(0008,0090)=LuVoX" ${pdfcdcm}
    echo "Setting ReferringPhysicianName: LuVoX"
    # set Manufacturer
    dcmodify -i "(0008,0070)=LuVoX by healthcAIr" ${pdfcdcm}
    echo "Setting Manufacturer: LuVoX by healthcAIr"


    # set StudyDate
    StudyDate=$(dcmdump +P "0008,0020" ${dcmscaffold} | grep -E -o "\[.*\]")
    StudyDate=${StudyDate:1:${#StudyDate}-2}
    dcmodify -i "(0008,0020)=${StudyDate}" ${pdfcdcm}
    echo "Setting StudyDate: ${StudyDate}"

    # set ContentDate
    ContentDate=$(dcmdump +P "0008,0023" ${dcmscaffold} | grep -E -o "\[.*\]")
    ContentDate=${ContentDate:1:${#ContentDate}-2}
    dcmodify -i "(0008,0023)=${ContentDate}" ${pdfcdcm}
    echo "Setting ContentDate: ${ContentDate}"

    # set AcquisitionDateTime
    AcquisitionDateTime=$(dcmdump +P "0008,002a" ${dcmscaffold} | grep -E -o "\[.*\]")
    AcquisitionDateTime=${AcquisitionDateTime:1:${#AcquisitionDateTime}-2}
    dcmodify -i "(0008,002a)=${AcquisitionDateTime}" ${pdfcdcm}
    echo "Setting AcquisitionDateTime: ${AcquisitionDateTime}"

    # set StudyTime
    StudyTime=$(dcmdump +P "0008,0030" ${dcmscaffold} | grep -E -o "\[.*\]")
    StudyTime=${StudyTime:1:${#StudyTime}-2}
    dcmodify -i "(0008,0030)=${StudyTime}" ${pdfcdcm}
    echo "Setting StudyTime: ${StudyTime}"

    # set ContentTime
    ContentTime=$(dcmdump +P "0008,0033" ${dcmscaffold} | grep -E -o "\[.*\]")
    ContentTime=${ContentTime:1:${#ContentTime}-2}
    dcmodify -i "(0008,0033)=${ContentTime}" ${pdfcdcm}
    echo "Setting ContentTime: ${ContentTime}"

    # set AccessionNumber
    AccessionNumber=$(dcmdump +P "0008,0050" ${dcmscaffold} | grep -E -o "\[.*\]")
    AccessionNumber=${AccessionNumber:1:${#AccessionNumber}-2}
    dcmodify -i "(0008,0050)=${AccessionNumber}" ${pdfcdcm}
    echo "Setting AccessionNumber: ${AccessionNumber}"

    # set StudyID
    StudyID=$(dcmdump +P "0020,0010" ${dcmscaffold} | grep -E -o "\[.*\]")
    StudyID=${StudyID:1:${#StudyID}-2}
    dcmodify -i "(0020,0010)=${StudyID}" ${pdfcdcm}
    echo "Setting StudyID: ${StudyID}"

    # delete pdf files
    find ${d} -type f -iname "*.pdf" -delete
    # storescu
    storescu -v --no-halt --scan-directories -aet DV-RAD-MLEARN -aec SECTRA_TELEMED 10.10.31.221 24500 "${d}/" &>> "${LUVOX_PREFIX}.log"
done

rm -rf "$LUVOX_PREFIX"
echo "Removed prefix path: $LUVOX_PREFIX"

rm -rf "$1"
echo "Removed input path: $1"

