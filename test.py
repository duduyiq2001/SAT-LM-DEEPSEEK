from z3 import *

# Variable definitions from signature
Students, (George, Helen, Irving, Kyle, Lenore, Nina, Olivia, Robert) = EnumSort('Students', ['George', 'Helen', 'Irving', 'Kyle', 'Lenore', 'Nina', 'Olivia', 'Robert'])
Days, (Monday, Tuesday, Wednesday) = EnumSort('Days', ['Monday', 'Tuesday', 'Wednesday'])
Times, (Morning, Afternoon) = EnumSort('Times', ['Morning', 'Afternoon'])
schedule = Function('schedule', Days, Times, Students)
gives_report = Function('gives_report', Students, BoolSort())

def check_option(assignments):
    s = Solver()
    # 1. Each slot is assigned to exactly one student, and each student gives at most one report
    all_students = [George, Helen, Irving, Kyle, Lenore, Nina, Olivia, Robert]
    all_days = [Monday, Tuesday, Wednesday]
    all_times = [Morning, Afternoon]
    # Build mapping: (day, time) -> student
    slot_vars = {}
    for (d, t), stu in assignments.items():
        slot_vars[(d, t)] = stu
        # Each slot is assigned to a student
        s.add(gives_report(stu))
        s.add(schedule(d, t) == stu)
    # Each student gives at most one report
    for stu in all_students:
        count = Sum([If(assignments[(d, t)] == stu, 1, 0) for d in all_days for t in all_times])
        s.add(If(gives_report(stu), count == 1, count == 0))
    # Exactly 6 students give reports
    s.add(Sum([If(gives_report(stu), 1, 0) for stu in all_students]) == 6)
    # 2. Tuesday is the only day George can give a report
    for d in all_days:
        for t in all_times:
            if assignments[(d, t)] == George and d != Tuesday:
                s.add(False)
    # 3. Neither Olivia nor Robert can give an afternoon report
    for d in all_days:
        if assignments[(d, Afternoon)] == Olivia or assignments[(d, Afternoon)] == Robert:
            s.add(False)
    # 4. If Nina gives a report, then on the next day Helen and Irving must both give reports, unless Nina's report is on Wednesday
    nina_slot = None
    for d in all_days:
        for t in all_times:
            if assignments[(d, t)] == Nina:
                nina_slot = (d, t)
    if nina_slot is not None:
        nina_day = nina_slot[0]
        if nina_day != Wednesday:
            # Next day
            if nina_day == Monday:
                next_day = Tuesday
            elif nina_day == Tuesday:
                next_day = Wednesday
            else:
                next_day = None
            # On next day, both Helen and Irving must give reports
            if next_day is not None:
                s.add(Or(assignments[(next_day, Morning)] == Helen, assignments[(next_day, Afternoon)] == Helen))
                s.add(Or(assignments[(next_day, Morning)] == Irving, assignments[(next_day, Afternoon)] == Irving))
    # 5. No student gives more than one report
    for stu in all_students:
        slots = [(d, t) for d in all_days for t in all_times if assignments[(d, t)] == stu]
        if len(slots) > 1:
            s.add(False)
    # 6. All slots are filled by unique students
    assigned_students = [assignments[(d, t)] for d in all_days for t in all_times]
    if len(set(assigned_students)) != 6:
        s.add(False)
    # 7. Only students in the list can be assigned
    for stu in assigned_students:
        if stu not in all_students:
            s.add(False)
    return s.check() == sat

def check_valid():
    # Parse each option into assignments: (day, time) -> student
    # (A) Mon. morning: Helen; Mon. afternoon: Robert
    #     Tues. morning: Olivia; Tues. afternoon: Irving
    #     Wed. morning: Lenore; Wed. afternoon: Kyle
    option_A = {
        (Monday, Morning): Helen,
        (Monday, Afternoon): Robert,
        (Tuesday, Morning): Olivia,
        (Tuesday, Afternoon): Irving,
        (Wednesday, Morning): Lenore,
        (Wednesday, Afternoon): Kyle,
    }
    # (B) Mon. morning: Irving; Mon. afternoon: Olivia
    #     Tues. morning: Helen; Tues. afternoon: Kyle
    #     Wed. morning: Nina; Wed. afternoon: Lenore
    option_B = {
        (Monday, Morning): Irving,
        (Monday, Afternoon): Olivia,
        (Tuesday, Morning): Helen,
        (Tuesday, Afternoon): Kyle,
        (Wednesday, Morning): Nina,
        (Wednesday, Afternoon): Lenore,
    }
    # (C) Mon. morning: Lenore; Mon. afternoon: Helen
    #     Tues. morning: George; Tues. afternoon: Kyle
    #     Wed. morning: Robert; Wed. afternoon: Irving
    option_C = {
        (Monday, Morning): Lenore,
        (Monday, Afternoon): Helen,
        (Tuesday, Morning): George,
        (Tuesday, Afternoon): Kyle,
        (Wednesday, Morning): Robert,
        (Wednesday, Afternoon): Irving,
    }
    # (D) Mon. morning: Nina; Mon. afternoon: Helen
    #     Tues. morning: Robert; Tues. afternoon: Irving
    #     Wed. morning: Olivia; Wed. afternoon: Lenore
    option_D = {
        (Monday, Morning): Nina,
        (Monday, Afternoon): Helen,
        (Tuesday, Morning): Robert,
        (Tuesday, Afternoon): Irving,
        (Wednesday, Morning): Olivia,
        (Wednesday, Afternoon): Lenore,
    }
    # (E) Mon. morning: Olivia; Mon. afternoon: Nina
    #     Tues. morning: Irving; Tues. afternoon: Helen
    #     Wed. morning: Kyle; Wed. afternoon: George
    option_E = {
        (Monday, Morning): Olivia,
        (Monday, Afternoon): Nina,
        (Tuesday, Morning): Irving,
        (Tuesday, Afternoon): Helen,
        (Wednesday, Morning): Kyle,
        (Wednesday, Afternoon): George,
    }
    options = {
        "A": option_A,
        "B": option_B,
        "C": option_C,
        "D": option_D,
        "E": option_E,
    }
    for letter, assignment in options.items():
        if check_option(assignment):
            return letter
    return None

if __name__ == "__main__":
    result = check_valid()
    print(result)