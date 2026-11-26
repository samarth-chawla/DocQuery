from pathlib import Path

import pymupdf

PAGES = {
    "Orbit Employee Handbook.pdf": [
        [
            "Paid time off",
            "Full-time employees receive 20 days of paid vacation per calendar year.",
            "Unused vacation of up to five days rolls over into the next year.",
        ],
        [
            "Working remotely",
            "Employees may work remotely up to three days per week.",
            "Employees must reserve a desk before visiting the Nairobi office.",
        ],
    ],
    "Orbit Travel Policy.pdf": [
        [
            "Business travel reimbursement",
            "Submit receipts within 30 calendar days of returning from a business trip.",
            "The daily meal allowance for overnight business travel is USD 60.",
        ]
    ],
}


def main() -> None:
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    for name, pages in PAGES.items():
        with pymupdf.open() as pdf:
            for lines in pages:
                page = pdf.new_page()
                page.insert_text((72, 72), "\n".join(lines), fontsize=12)
            path = output / name
            pdf.save(path)
            print(path)


if __name__ == "__main__":
    main()
