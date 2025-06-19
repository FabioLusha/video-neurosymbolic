# STAR Benchmark

## Schema

- question_id: string
- question:    string
- video_id:    string
- start:       float
- end:         float
- answer:      string
- qustion_program: array<object>
    items: {
          "properties": {
              "function": {"type": "string"}
              "value_input": array<string>
          }
      }
- choices: array<object>
    items: {
          "properties": {
              "choice_id": int,
              "choice": string
              "choice_program": like question_program
          }
    }
- situations: dict<string, object>
    key: string
    value: object {
        "properties": {
            "rel_pairs": array<array of 2 strings>
            "rel_labels": array<string>
            "actions": array<string>
            "bbox": <array<array<float>>
            "bbox_labels": <array<string>
        }
    }

## Desription

**question_id**
...
**video_id, start, end**
STAR Benchmark piggybacks the Charades video and 
