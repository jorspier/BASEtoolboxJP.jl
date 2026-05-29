using XLSX
path = "examples/Data/sektorale-und-gesamtwirtschaftliche-vermoegensbilanzen-xls-data.xlsx"
xf = XLSX.readxlsx(path)
println("Sheets: ", XLSX.sheetnames(xf))
for sh in ["S1+S11", "S12+S13", "S14+S15"]
    ws = xf[sh]
    println("\n--- $sh ---")
    for r in 1:10
        row = [ws[r, c] for c in 1:29]
        println(row)
    end
end
