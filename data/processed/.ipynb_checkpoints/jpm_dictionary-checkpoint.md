| Field           | Description                                                                                   |
|-----------------|-----------------------------------------------------------------------------------------------|
| section         | Whether the row belongs to the prepared remarks (`presentation`) or the Q&A (`qa`) section.   |
| question_number | Sequential index of the analyst question within the Q&A (blank for presentation rows).        |
| answer_number   | Sequential index of the executive answer tied to a question (blank for presentation rows).    |
| speaker_name    | Name of the individual speaking (e.g., “Jeremy Barnum”, “Jamie Dimon”, “Christopher McGratty”).|
| role            | Speaker’s role or title (e.g., “Chief Financial Officer”, “Chairman & CEO”, “Analyst”).       |
| company         | Organization the speaker represents (e.g., JPMorgan Chase for executives, brokerage/firm for analysts). |
| content         | The spoken text content (paragraph of remarks, question, or answer).                          |
| year            | Calendar year of the earnings call.                                                           |
| quarter         | Quarter of the earnings call (e.g., Q1, Q2).                                                  |
| is_pleasantry   | Boolean flag (True/False) marking polite exchanges such as greetings, thanks, or small talk.  |
| source_pdf      | Filename of the transcript PDF the row was extracted from.                                    |
