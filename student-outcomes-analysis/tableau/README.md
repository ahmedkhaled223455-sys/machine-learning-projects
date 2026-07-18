# Tableau Dashboard — Build Guide

A fourth engine for the same business question this project already
answers three ways (SQLite, Snowflake, PySpark): does test prep move
scores, and who's actually taking it? This one is a BI dashboard, built
by hand in Tableau Public's free browser-based editor — no desktop
install needed, no code. That's genuinely how this step works: unlike
the SQL/Spark ports, there's no way to script a real Tableau workbook
from a terminal, so this is a walkthrough for you to follow yourself
(~20 minutes). Send me the published link when you're done and I'll
wire it into both READMEs.

Data file: [`students_dashboard_data.csv`](students_dashboard_data.csv)
— the same 1,000 students as the rest of the project, columns
snake_cased and category labels cleaned up (`Completed` / `Not
Completed`, `Standard` / `Free/Reduced`) so they read well on a chart
without any relabeling in Tableau.

## 0. Setup

1. Go to [public.tableau.com](https://public.tableau.com) and create a
   free account if you don't have one (top right → **Sign Up**).
2. Sign in, then click **Create a Viz** (or **Create** → **Web
   Authoring** from the top nav).
3. **Connect to Data** → **Upload from Computer** → select
   `students_dashboard_data.csv` from this folder.
4. Tableau shows a data-source preview. Check the row count reads
   **1,000**. Click **Sheet 1** (bottom tab) to start building.

You'll now see: the **Data pane** on the left (fields split into
*Dimensions*, text/category fields in blue, and *Measures*, numeric
fields in green), the **Columns**/**Rows** shelves along the top of the
canvas, and the **Marks card** to the left of the canvas (Color, Size,
Label, Detail, Tooltip). Field names display with underscores turned
into spaces and title-cased automatically — `math_score` shows as
**Math Score**.

## 1. Sheet: "Avg Math Score by Test Prep"

1. Drag **Test Prep** from Dimensions onto the **Columns** shelf.
2. Drag **Math Score** from Measures onto the **Rows** shelf. It
   appears as `SUM(Math Score)` and draws two tall bars.
3. Click the dropdown arrow on that pill (on the Rows shelf) →
   **Measure** → **Average**. Bars now show average, not sum — this is
   the same number as `avg_math` in `results/sql_results.md`.
4. Drag **Test Prep** from Dimensions onto **Color** on the Marks card,
   so the two bars get distinct colors.
5. Click the **Label** button on the Marks card → check **Show mark
   labels**, so each bar displays its value.
6. Double-click the "Sheet 1" tab at the bottom and rename it to
   `Avg Math Score by Test Prep`.

*Optional polish:* drag **Reading Score**, then **Writing Score**, onto
the same axis area on the canvas until a thin green line appears at the
right edge of the existing axis, then drop — this merges all three into
one grouped bar chart via Tableau's automatic **Measure Names** /
**Measure Values** fields. Set each pill's aggregation to Average the
same way as step 3.

## 2. Sheet: "Score Distribution by Test Prep"

1. Click the **+** tab at the bottom to add a new sheet.
2. In the Data pane, click **Test Prep**, then Ctrl-click (Cmd-click on
   Mac) **Math Score** to select both.
3. In the **Show Me** panel (top right, click to expand if collapsed),
   click the **box-and-whisker plot** icon. Tableau builds it
   automatically.
4. Rename the tab to `Score Distribution by Test Prep`.

This shows the full spread, not just the average — a fairer picture
than a bar chart alone, and a natural thing to point out in an
interview.

## 3. Sheet: "% At-Risk by Test Prep"

This reproduces the `pct_below_60` numbers from `sql_results.md`
(24.3% for completers, 36.8% for non-completers) as a calculated field.

1. Right-click any blank space in the Data pane → **Create Calculated
   Field...**
2. Name it `At Risk Flag`. Formula:
   ```
   IF [Math Score] < 60 THEN 1 ELSE 0 END
   ```
   (Type `[Math` and pick **Math Score** from the autocomplete dropdown
   to avoid a typo.) Click **OK**.
3. New sheet (**+** tab). Drag **Test Prep** to **Columns**.
4. Drag the new **At Risk Flag** measure to **Rows**. It shows
   `SUM(At Risk Flag)` — change it to **Average** the same way as
   section 1, step 3. Since the flag is 0/1, its average *is* the
   proportion at risk.
5. Right-click the axis → **Format...** → under **Numbers**, choose
   **Percentage**.
6. Rename the tab to `Pct At Risk by Test Prep`.

## 4. Sheet: "Prep Completion Rate by Parental Education"

Reproduces the selection-check query — prep completion varies a lot by
parental education (43% down to 28.6%), which is exactly why the SQL
project's regression adjusts for it.

1. Calculated field (same menu as above): name it
   `Completed Prep Flag`, formula:
   ```
   IF [Test Prep] = "Completed" THEN 1 ELSE 0 END
   ```
2. New sheet. Drag **Parental Education** to **Columns**, drag
   **Completed Prep Flag** to **Rows**, set its aggregation to
   **Average**, format the axis as **Percentage** (same steps as
   section 3).
3. Click the descending-sort icon in the toolbar (or right-click the
   axis → **Sort** → **Descending**) so bars run from highest to lowest
   completion rate.
4. Rename the tab to `Prep Completion by Parental Education`.

## 5. Assemble the dashboard

1. Click the **+** tab at the bottom, but this time pick **New
   Dashboard** (it's a second kind of "+", or use **Dashboard** menu →
   **New Dashboard**).
2. Set the dashboard size to **Automatic** (top of the left panel) so
   it adapts to different screens.
3. Drag all four sheets from the left panel onto the canvas — a 2×2
   grid reads well (drag each sheet near a corner/edge until you see
   the gray drop zone outline).
4. Drag **Test Prep**, **Gender**, and **Parental Education** from any
   one sheet's card onto the dashboard as filters: click a sheet on the
   dashboard, click the funnel icon that appears top-right of it for
   the field you want to filter by, or drag the field from the Data
   pane directly onto the dashboard canvas and choose **Filter**. Set
   each filter to **Apply to Worksheets → All Using This Data Source**
   so one filter click updates every chart at once.
5. Add a title: double-click the dashboard title placeholder at the
   top, type something like `Does Test Prep Work? — Student Outcomes
   Dashboard`.
6. Click **File** → **Save** (or the Save icon). On Tableau Public,
   saving *is* publishing — it goes straight to your public profile.

## 6. Get the link

After saving, Tableau opens your published viz's page on
`public.tableau.com`. Copy that URL — that's your shareable dashboard
link (and there's a **Share** button for an embed code, if you ever
want it on a personal site).

Send me that link and I'll add it to this project's README and the
top-level portfolio README, next to the SQLite/Snowflake/PySpark
mentions of the same analysis.
