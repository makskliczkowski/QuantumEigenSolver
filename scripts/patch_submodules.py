import os
import sys

def fix_statistical_h():
    filepath = "cpp/library/source/src/Include/statistical.h"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, "r") as f:
        content = f.read()

    # The block to remove (exact match required, copy-pasted from original file structure)
    to_remove = """	template<>
	void setHistogramCounts(const arma::Col<std::complex<double>>& _values, bool _setBins)
	{
		// get the bin edges - those are determined by the minimum and maximum values
		arma::Col<double> _valuesr = arma::real(_values);
		this->setHistogramCounts(_valuesr, _setBins);
	}

	template<>
	void setHistogramCounts(const arma::subview_col<std::complex<double>>& _values, bool _setBins)
	{
		// get the bin edges - those are determined by the minimum and maximum values
		arma::Col<double> _valuesr = arma::real(_values);
		this->setHistogramCounts(_valuesr, _setBins);
	}"""

    if to_remove in content:
        print(f"Patching {filepath}: Removing in-class specializations.")
        content = content.replace(to_remove, "")

        # Append the fixed specializations to the end of the file
        to_append = """
// PATCHED BY CI
template<>
inline void Histogram::setHistogramCounts(const arma::Col<std::complex<double>>& _values, bool _setBins)
{
	// get the bin edges - those are determined by the minimum and maximum values
	arma::Col<double> _valuesr = arma::real(_values);
	this->setHistogramCounts(_valuesr, _setBins);
}

template<>
inline void Histogram::setHistogramCounts(const arma::subview_col<std::complex<double>>& _values, bool _setBins)
{
	// get the bin edges - those are determined by the minimum and maximum values
	arma::Col<double> _valuesr = arma::real(_values);
	this->setHistogramCounts(_valuesr, _setBins);
}
"""
        # Ensure we don't append twice
        if "// PATCHED BY CI" not in content:
            content += to_append

        with open(filepath, "w") as f:
            f.write(content)
        print("Patch applied successfully.")
    else:
        print(f"Block not found in {filepath}. Assuming already patched or different content.")
        # If the removal block is missing, check if the append block is there.
        # If not, we might be in a state where the file is fresh but whitespace differs?
        # But we rely on exact content match.
        # For the purpose of this CI fix, if exact match fails, we warn but proceed.

if __name__ == "__main__":
    print("Running submodule patches...")
    fix_statistical_h()
